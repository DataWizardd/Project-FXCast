import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json, random
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

TRAIN_JSON = "train_original.json"
VALID_JSON = "valid_original.json"

SAMPLE_N = 500
SEED = 42
BATCH_SIZE = 8
GEN_KW = dict(max_length=80, min_length=10, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3)

MODELS = [
    ("csebuetnlp/mT5_multilingual_XLSum", "mt5_xlsum"),   
    ("eenzeenee/t5-base-korean-summarization", "t5_ko_sum"),
    ("EbanLee/kobart-summary-v3", "kobart_sum_v3"),
]

def set_seed(seed=SEED):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _to_str(x):
    if x is None: return ""
    if isinstance(x, list): return " ".join(_to_str(e) for e in x)
    if isinstance(x, dict): return " ".join(_to_str(v) for v in x.values())
    return str(x)

CONTENT_KEYS = ["document","text","content","article","article_original","orginal_text","news","body","doc","source"]
REF_KEYS_ABS = ["abstractive","abstract","reference","target_summary","summaries"]
REF_KEYS_SUM = ["summary","summ","target","headline"]
REF_KEYS_EXT = ["extractive","extract_summary"]

def _pick_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d and d[k] is not None:
            return k
    return None

def _collect_pairs(node: Any, pairs: List[Tuple[str,str]]):
    if isinstance(node, dict):
        ckey = _pick_key(node, CONTENT_KEYS)
        rkey = _pick_key(node, REF_KEYS_ABS) or _pick_key(node, REF_KEYS_SUM) or _pick_key(node, REF_KEYS_EXT)
        if ckey and rkey:
            c = _to_str(node.get(ckey, "")).replace("\n"," ").strip()
            r = _to_str(node.get(rkey, "")).replace("\n"," ").strip()
            if c and r: pairs.append((c, r))
        for v in node.values():
            _collect_pairs(v, pairs)
    elif isinstance(node, list):
        for item in node: _collect_pairs(item, pairs)

def robust_load_aihub(*paths) -> pd.DataFrame:
    all_pairs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        _collect_pairs(data, all_pairs)
    cleaned, seen = [], set()
    for c, r in all_pairs:
        key = (hash(c), hash(r))
        if key in seen: continue
        seen.add(key); cleaned.append((c,r))
    if not cleaned:
        raise ValueError("원문/요약 쌍을 찾지 못했습니다.")
    return pd.DataFrame(cleaned, columns=["content","reference"])

def build_inputs(text: str, model_id: str) -> str:
    return f"다음 한국어 뉴스 기사를 한두 문장으로 요약하세요.\n본문: {text}" if model_id in ["mt5_xlsum","t5_ko_sum"] else text

def generate(model_name, model_id, texts, gen_kw=GEN_KW, batch_size=BATCH_SIZE, device=None):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); model.eval()
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating [{model_id}]"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=gen_kw["max_length"],
                min_length=gen_kw["min_length"],
                num_beams=gen_kw["num_beams"],
                length_penalty=gen_kw["length_penalty"],
                no_repeat_ngram_size=gen_kw["no_repeat_ngram_size"]
            )
        dec = tok.batch_decode(out, skip_special_tokens=True)
        preds.extend([d.replace("\n"," ").strip() for d in dec])
    return preds

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    score_list = [scorer.score(ref, pred) for pred, ref in zip(preds, refs)]
    mean = lambda k: float(np.mean([s[k].fmeasure for s in score_list]))
    return {"rouge1": round(mean("rouge1"),4), "rouge2": round(mean("rouge2"),4), "rougeL": round(mean("rougeL"),4)}

def main():
    set_seed(SEED)
    df = robust_load_aihub(TRAIN_JSON, VALID_JSON)
    if SAMPLE_N and SAMPLE_N < len(df):
        df = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
    print("샘플 수:", len(df))
    results = []
    for model_name, model_id in MODELS:
        inputs = [build_inputs(t, model_id) for t in df["content"].tolist()]
        preds = generate(model_name, model_id, inputs)
        scores = compute_rouge(preds, df["reference"].tolist())
        print(f"[{model_id}] → {scores}")
        df.assign(generated=preds).to_csv(f"samples_{model_id}.csv", index=False, encoding="utf-8-sig")
        results.append({"model": model_id, **scores})
    pd.DataFrame(results).to_csv("metrics_rouge.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()