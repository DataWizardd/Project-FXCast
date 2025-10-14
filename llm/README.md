# FX News NLP Pipeline

환율(FX) 관련 뉴스를 대상으로 전처리 → 카테고리 분류(직접/간접/무관) → 요약 → 감정 분석까지 수행하는 경량 파이프라인입니다. OpenAI GPT-4o-mini를 사용합니다.

## 구성요소

- `fx_news_preprocess_and_summarize.py`  
  - 불용어 제거 및 텍스트 정규화
  - 기사 본문 요약(LLM)

- `fx_news_category_labeler.py`  
  - 환율 관련성: `Direct / Indirect / None` 분류(LLM)

- `fx_news_sentiment.py`  
  - 요약문을 대상으로 감정 분석(`pos/neu/neg`)
