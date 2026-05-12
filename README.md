# Credit_Score_DeepLearning_Competition

### 1. 프로젝트명
신용 점수 분류 딥러닝 모델 개발 및 성능 개선

### 2. 기간
2026년 5월 12일

### 3. 기술스택: 라이브러리 명시
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-learn (sklearn)
- Boosting Algorithms: XGBoost, LightGBM
- Deep Learning: PyTorch
- Visualization: Matplotlib, Seaborn

### 4. 데이터 출처
수업 제공 신용 점수 데이터셋 (train2.csv)

### 5. 데이터 전처리
1) 개인 식별 컬럼 제거 (ID, Customer_ID, Name, SSN) : 신용 점수와 통계적 상관관계가 없고, 모델이 해당 값에 과적합될 위험이 있어 제거
2) y컬럼 선정 (Credit_Score) : 고유값 2~10개 기준으로 타겟 후보를 추린 뒤, 소득·부채·행동 패턴 등 다수의 독립 변수가 결합돼 도출되는 최종 결과값인 Credit_Score를 타겟으로 선정
3) Ordinal Encoding 적용 : Good=2, Standard=1, Poor=0으로 등급 순서를 수치화해 모델이 순서 관계를 학습할 수 있도록 처리
4) 상관관계 기반 컬럼 제거 : 히트맵을 통해 Credit_Score와 상관계수가 0에 가까운 Month, Credit_Utilization_Ratio를 추가 제거해 노이즈 감소
5) 파생변수 6개 생성 : DTI, EMI_to_Salary, Pure_Free_Cash, Age_Group, Loan_Per_Account, Investment_Ratio (기존 컬럼과 중복 없이 설계)
6) 결측값 처리 : Credit_Score 결측 행 dropna로 제거

### 6. EDA 및 해석

> 수치형 컬럼과 Credit_Score 간 상관관계 히트맵

![상관관계 히트맵](이미지_경로_입력)

- Credit_Score와 상관계수가 유의미한 컬럼을 선별하고, 0에 가까운 변수(Month, Credit_Utilization_Ratio)를 제거
- 파생변수(DTI, EMI_to_Salary 등)를 추가해 신용 위험을 다각도로 수치화

> 타겟 후보 시각화 (Unique Value Count 기준 막대 그래프)

![타겟 후보 그래프](이미지_경로_입력)

- 고유값 2~10개 조건으로 후보 컬럼을 추출하고, Credit_Score가 분류 문제에 가장 적합한 타겟임을 시각적으로 확인

### 7. Feature Selection & 모델링 & 튜닝

> 모델 비교 결과

![모델 비교 결과](이미지_경로_입력)

- RandomForest / XGBoost / LightGBM 세 모델을 동일 조건에서 비교, F1-Score(macro) 기준 RandomForest가 최고 성능 기록
- RandomForest의 `feature_importances_` 기반으로 중요도 하위 5% 피처 제거 → 노이즈 감소 및 과적합 방지
- 최종 모델로 **SuperMLP (PyTorch 기반 다층 퍼셉트론)** 설계
  - BatchNorm + Dropout(0.4 / 0.2) 적용으로 학습 안정화 및 과적합 억제
  - 옵티마이저: **AdamW** (weight_decay=1e-5) — 일반 Adam 대비 L2 정규화 강화
  - 스케줄러: **OneCycleLR** — 초반 학습률을 높여 Local Minima 탈출, 후반 수렴 유도

### 8. 성능 결과

> Validation Accuracy 학습 곡선

![학습 곡선](이미지_경로_입력)

| 구분 | 내용 |
|---|---|
| 평가 지표 | Accuracy, F1-Score (macro) |
| 모니터링 방식 | 10 epoch마다 Valid Acc 출력 + Best Acc 갱신 |
| 목표 기준 | Valid Score 75 이상 |

```
[epoch] Train Loss: x.xxxx | Valid Acc: x.xxxx | Best Acc: x.xxxx
```

### 9. 개선 과정 요약
- 1차: Adam(lr=0.0003), epoch 100 → 학습률 낮아 수렴 부족
- 2차: LabelEncoder 추가로 CUDA 오류 방지, AdamW + OneCycleLR 도입, batch 512, epoch 300으로 확장
- Best Accuracy 기준 모니터링 구조로 최고 성능 지점 추적

### 10. Reference
1) 사용 라이브러리 및 공식 문서
   - PyTorch: 딥러닝 모델(SuperMLP) 구현 및 GPU 학습
   - Scikit-learn: 데이터 전처리(LabelEncoder, train_test_split) 및 머신러닝 모델 비교
   - XGBoost / LightGBM: 트리 기반 부스팅 모델 성능 비교
   - Seaborn / Matplotlib: 상관관계 히트맵 및 타겟 후보 시각화

2) 분석 방법론
   - EDA (탐색적 데이터 분석): 상관관계 분석 및 고유값 기반 타겟 선정
   - Feature Engineering: 파생변수 6개 생성으로 예측력 강화
   - Feature Selection: RandomForest 중요도 기반 하위 5% 제거
   - Deep Learning Tuning: AdamW + OneCycleLR + BatchNorm + Dropout 조합으로 성능 개선
