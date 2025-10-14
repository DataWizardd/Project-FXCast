# FX News NLP Pipeline

OpenAI GPT-4o-mini를 사용

## 구성요소

- `fx_news_preprocess_and_summarize.py`  
  - 불용어 제거 및 텍스트 정규화
  - 기사 본문 요약(LLM)

- `fx_news_category_labeler.py`  
  - 환율 관련성: `Direct / Indirect / None` 분류(LLM)

- `fx_news_sentiment.py`  
  - 요약문을 대상으로 감정 분석(`pos/neu/neg`)
