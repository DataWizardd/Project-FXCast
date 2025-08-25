import os, re, json, random
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# =========================
# 환경 설정
# =========================
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
client = OpenAI(api_key=OPENAI_API_KEY)

SRC = "naver_finance_news_2020_2024_with_kobart_v2.csv"  # 입력 CSV
DST = "naver_finance_news_labeled_gpt_v2.csv"            # 결과 CSV
MODEL = "gpt-4o-mini"
TRUNCATE_CHARS = 1200
SEED = 42
random.seed(SEED)

SENTIMENT_LABELS = ["긍정", "부정", "중립"]

# =========================
# Few-shot 구성
# =========================
# 1) 코드 내 기본 예시 (한국 원/달러 관점에 맞춘 예시)
DEFAULT_FEWSHOTS = [
    # 긍정: 원/달러 하락(원화 강세), 위험선호 확대, 달러 약세
    {
        "text": "달러화가 약세를 보이며 원/달러 환율이 10원 하락했다. 외국인 주식 순매수도 확대됐다.",
        "label": "긍정",
        "rationale": "달러 약세·환율 하락은 원화 강세 요인"
    },
    {
        "text": "미국 물가가 둔화했다는 지표가 발표되며 위험선호가 회복, 신흥국 통화가 강세를 보였다.",
        "label": "긍정",
        "rationale": "완화적 환경 신호로 위험선호 확대"
    },

    # 부정: 원/달러 상승(원화 약세), 위험회피 확대, 달러 강세·긴축
    {
        "text": "연준의 추가 긴축 경계감에 달러 인덱스가 급등, 원/달러 환율이 20원 뛰었다.",
        "label": "부정",
        "rationale": "달러 강세·환율 상승은 원화 약세"
    },
    {
        "text": "중동 긴장 고조로 안전자산 선호가 확대되며 엔화·달러가 강세, 원화는 약세를 보였다.",
        "label": "부정",
        "rationale": "위험회피 확대는 원화 약세 요인"
    },

    # 중립: 방향 판단 어려움·정보 위주
    {
        "text": "기획재정부는 외환 건전성 규정 개편안을 다음 달 발표할 예정이라고 밝혔다.",
        "label": "중립",
        "rationale": "정책 일정 소개로 방향성 언급 없음"
    },
    {
        "text": "서울 외환시장의 점심시간 거래량이 전일과 유사한 수준을 기록했다.",
        "label": "중립",
        "rationale": "사실 전달 위주로 뚜렷한 방향성 부재"
    },
]

# 2) 외부 few-shot 파일(.jsonl or .json)에서 추가 로드 
FEWSHOT_PATH = os.getenv("FEWSHOT_PATH", "")  
FEWSHOT_K = int(os.getenv("FEWSHOT_K", "4"))  

def load_fewshots() -> list:
    few = list(DEFAULT_FEWSHOTS)
    if FEWSHOT_PATH and os.path.exists(FEWSHOT_PATH):
        ext = os.path.splitext(FEWSHOT_PATH)[1].lower()
        try:
            if ext == ".jsonl":
                with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        few.append(obj)
            elif ext == ".json":
                with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
                    few.extend(json.load(f))
        except Exception as e:
            print(f"[경고] FEWSHOT_PATH 로드 실패: {e}")
    # 라벨 검증 및 클린업
    cleaned = []
    for ex in few:
        t = str(ex.get("text", "")).strip()
        l = ex.get("label", "")
        r = str(ex.get("rationale", "")).strip()
        if t and l in SENTIMENT_LABELS:
            cleaned.append({"text": t, "label": l, "rationale": r})
    return cleaned

FEWSHOTS = load_fewshots()

# =========================
# 프롬프트 세팅 (few-shot)
# =========================
SYSTEM_PROMPT = (
    "너는 한국 원/달러 환율 관련 금융 뉴스의 감성을 분류하는 어시스턴트다.\n"
    "- 긍정: 원/달러 하락(원화 강세), 위험선호 확대, 달러 약세, 완화적 환경\n"
    "- 부정: 원/달러 상승(원화 약세), 위험회피 확대, 달러 강세, 긴축적 환경\n"
    "- 중립: 방향성 판단이 어려움, 사실·정보 위주\n"
    "반드시 구조화된 JSON만 반환하라."
)

def build_fewshot_messages(target_text: str, k: int = FEWSHOT_K):
    """few-shot 예시 k개를 무작위로 섞어 user 메시지에 포함"""
    # 텍스트 과다 방지
    target_text = (target_text or "")[:TRUNCATE_CHARS].strip()
    # 예시 샘플링(라벨 다양성 확보를 위해 셔플)
    ex = FEWSHOTS[:]
    random.shuffle(ex)
    ex = ex[:max(0, k)]

    # 예시 블록을 하나의 문자열로 구성
    examples_block = []
    for i, e in enumerate(ex, 1):
        examples_block.append(
            f"[예시{i}]\n텍스트: {e['text']}\n라벨: {e['label']}\n근거: {e.get('rationale','')}".strip()
        )
    examples_str = "\n\n".join(examples_block) if examples_block else "(예시 없음)"

    user_prompt = (
        "다음 텍스트의 감성을 판단하라. 텍스트 근거 기반으로만 판단하고 과잉 추론은 금지한다.\n\n"
        f"{examples_str}\n\n"
        "===== 분류 대상 텍스트 =====\n"
        f"\"\"\"\n{target_text}\n\"\"\""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

# =========================
# JSON Schema 및 툴 정의 
# =========================
RESPONSES_JSON_SCHEMA = {
    "name": "SentimentLabel",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": SENTIMENT_LABELS},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string", "description": "라벨 선택 근거를 한국어 한두 문장으로 요약"}
        },
        "required": ["label", "confidence"],
        "additionalProperties": False
    },
    "strict": True
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "set_label",
        "description": "감성 라벨과 신뢰도, 근거를 구조화해 반환",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": SENTIMENT_LABELS},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string", "description": "라벨 선택 근거(한국어)"}
            },
            "required": ["label", "confidence"],
            "additionalProperties": False
        }
    }
}]

# =========================
# 유틸
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c).strip().lower() for c in df.columns]
    return df

def clean_text(x) -> str:
    if not isinstance(x, str):
        x = "" if pd.isna(x) else str(x)
    x = x.replace("\x00", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

TEXT_COL_CANDIDATES = ["summary_kobart_v3", "content", "title"]

def pick_text(row: pd.Series) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in row:
            txt = clean_text(row.get(c, ""))
            if txt:
                return txt
    return ""

class APIRetryableError(Exception):
    pass

@retry(
    reraise=True,
    retry=retry_if_exception_type(APIRetryableError),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5)
)
def classify_text(text: str) -> dict:
    """responses.create(+json_schema) → 실패 시 chat.completions(function calling) 폴백 (few-shot 포함)"""
    if not text:
        return {"label": "중립", "confidence": 0.0, "rationale": "빈 텍스트"}
    messages = build_fewshot_messages(text)

    # 1) responses.create (JSON Schema 강제)
    try:
        resp = client.responses.create(
            model=MODEL,
            input=messages,  
            response_format={"type": "json_schema", "json_schema": RESPONSES_JSON_SCHEMA},
            temperature=0
        )
        out = resp.output[0].content[0].text 
        data = json.loads(out)
        if data.get("label") not in SENTIMENT_LABELS:
            data["label"] = "중립"
        data["confidence"] = float(data.get("confidence", 0.0))
        data["rationale"] = data.get("rationale", "")
        return data

    except TypeError:
        # 환경 차이로 response_format 미지원 시 폴백
        pass
    except Exception as e:
        # 일시 오류 → 재시도
        raise APIRetryableError(str(e))

    # 2) chat.completions + function calling 폴백
    try:
        ch = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "set_label"}}
        )
        tool_calls = ch.choices[0].message.tool_calls
        if not tool_calls:
            return {"label": "중립", "confidence": 0.0, "rationale": "툴 호출 없음"}
        args_str = tool_calls[0].function.arguments or "{}"
        data = json.loads(args_str)
        if data.get("label") not in SENTIMENT_LABELS:
            data["label"] = "중립"
        data["confidence"] = float(data.get("confidence", 0.0))
        data["rationale"] = data.get("rationale", "")
        return data
    except Exception as e:
        raise APIRetryableError(str(e))

def main():
    df = pd.read_csv(SRC, encoding="utf-8-sig")
    df = clean_columns(df)

    # date → year 파생 (YYYY 또는 YYYYMMDD 모두 허용)
    if "year" not in df.columns:
        if "date" in df.columns:
            df["year"] = (
                df["date"].astype(str)
                .str[:4]
                .str.replace(r"\D", "", regex=True)
            )
        else:
            df["year"] = pd.NA

    # 라벨 컬럼 준비
    if "label" not in df.columns: df["label"] = ""
    if "label_confidence" not in df.columns: df["label_confidence"] = 0.0
    if "label_rationale" not in df.columns: df["label_rationale"] = ""

    # 입력 텍스트 선택
    df["_text_for_label"] = df.apply(pick_text, axis=1)

    if "year" in df.columns:
        print("=== 연도별 원본 개수 ===")
        print(df["year"].value_counts(dropna=False).sort_index())

    total = len(df)
    empty_cnt = int((df["_text_for_label"] == "").sum())
    if empty_cnt > 0:
        print(f"[경고] 라벨링 입력이 비어있는 행: {empty_cnt}개")

    labels, confs, rats = [], [], []
    for i, row in df.iterrows():
        res = classify_text(row["_text_for_label"])
        labels.append(res.get("label", "중립"))
        confs.append(res.get("confidence", 0.0))
        rats.append(res.get("rationale", ""))
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"{i+1}/{total} done")

    df["label"] = labels
    df["label_confidence"] = confs
    df["label_rationale"] = rats
    df.drop(columns=["_text_for_label"], inplace=True)

    df.to_csv(DST, index=False, encoding="utf-8-sig")
    print("\n저장 완료:", DST)

    if "year" in df.columns:
        print("\n=== 샘플 연도별 개수(라벨 부착 후) ===")
        print(df["year"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    main()
