# 📈 딥러닝 기반 원/달러 환율 예측  
### 글로벌 이벤트 & 뉴스 감성 지표의 활용

---

## 📝 연구 개요
본 연구는 글로벌 이벤트(GDELT)와 뉴스 감성 지표를 결합하여  
**USD/KRW 종가**를 예측하는 딥러닝 기반 프레임워크를 구축하는 데 목적이 있습니다.  

- **입력 피처 조합**  
  - 🟦 **Baseline** (거시/시장 데이터)  
  - 🟩 **+ GDELT** (글로벌 이벤트 지표)  
  - 🟨 **+ 뉴스 요약 감성**  
  - 🟧 **+ GDELT + 뉴스 요약 감성**

- **예측 모델**  
  - LSTM  
  - GRU  
  - CNN_LSTM  
  - CNN_GRU  

- **평가 지표**  
  - MSE  
  - MAPE  

- **하이퍼파라미터 조합**  
  - Lookback (5, 10, 20, 30, 60, 90)  
  - Shift (1일 예측)  
  - Seeds (42, 55, 68)  

---

## 📊 성능 비교

### ✅ Table A. 모델별 성능 (MSE 기준 정렬)
<p align="center">
  <img width="600" alt="model-wise-performance" src="https://github.com/user-attachments/assets/00599aa8-d42e-4af8-8a03-9fc993ec5baf" />
</p>

---

### ✅ Table B. 조건별 성능 (모델 평균)
<p align="center">
  <img width="450" alt="case-lookback-performance" src="https://github.com/user-attachments/assets/30e36748-e4f1-4754-b0ff-2da7d6562edc" />
</p>

---

## 📉 시각화 결과

### 1️⃣ 전체 구간 예측 (실제 vs 예측)
<p align="center">
  <img width="950" alt="actual-vs-predicted" src="https://github.com/user-attachments/assets/b98abe9d-03c6-441b-84e9-0b2d21c72a04" />
</p>

---

### 2️⃣ 변수 중요도 (Permutation Importance, Top-K)
<p align="center">
  <img width="950" alt="perm_importance_topk" src="https://github.com/user-attachments/assets/e399173a-8b69-442a-8a1e-8d78853fb7f7" />
</p>


### 3️⃣ 히트맵 (Correlation Heatmap)
<p align="center">
  <img width="820" alt="feature-corr-heatmap" src="https://github.com/user-attachments/assets/747bd093-bc56-4e2a-a44f-9a84a8240e8a" />
</p>


## ✨ 결론
- **뉴스 감성**과 **글로벌 이벤트 지표**는 베이스라인 대비 예측 성능 개선에 기여함.  
- 환율 예측에서 **비정형 데이터(뉴스, 이벤트)** 활용 가능성을 확인함.  
