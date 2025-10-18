# 모델 개발 보고서 개요

**대회**: 토스 NEXT ML CHALLENGE - 광고 클릭 예측(CTR) 모델 개발
**팀명**: [Your Team Name]
**작성일**: 2025년 10월 17일

---

## 📋 보고서 구성 (권장)

### 1. 개요 (Executive Summary)
- 대회 목표 및 평가 지표
- 최종 제출 모델 요약
- Private Score 및 순위
- 핵심 성과 및 기여도

### 2. 데이터 분석 (EDA)
- 데이터셋 구조 및 통계
  - Train: 10.7M rows, 124 features
  - Test: 2.8M rows
  - Class Imbalance: 1:51.6 (clicked:not clicked)

- 주요 발견사항
  - inventory_id의 높은 카디널리티 (특별 처리 필요)
  - Numerical features의 스케일 차이
  - Feature correlation 분석

- 피처 중요도 분석
  - XGBoost Feature Importance
  - DeepFM Embedding 분석

### 3. 피처 엔지니어링
- Categorical Features 처리
  - Label Encoding
  - inventory_id 최적화 (10차원 embedding)

- Numerical Features 처리
  - QuantileTransformer (정규분포 변환)
  - Clipping (outlier 처리)

- 추가 피처 없음 (원본 피처만 사용)

### 4. 모델 아키텍처

#### 4.1 XGBoost v6
```
- 알고리즘: Gradient Boosting Decision Tree
- 주요 설정:
  * max_depth: 9
  * learning_rate: 0.039
  * n_estimators: 280
  * Regularization: L1=0.27, L2=0.40
  * scale_pos_weight: 51.6
```

**특징**:
- Tree-based 모델의 강점 (피처 interaction)
- 강력한 정규화로 과적합 방지
- Class imbalance 처리

**성능**:
- CV Score: [Your Score]
- Public LB: [Your Score]

#### 4.2 DeepFM v2.2
```
- 아키텍처: FM + DNN
- FM Part: 2차 피처 interaction
- Deep Part: [256, 128] hidden units
- Embedding: inventory_id (10D), others (16D)
- Optimization: AdamW, LR=0.001
```

**특징**:
- 명시적 + 암묵적 interaction 학습
- Embedding 최적화로 파라미터 효율성
- Dropout 0.25, Weight Decay 2e-5

**성능**:
- OOF Score: 0.3507
- Public LB: 0.3466

### 5. 학습 전략

#### 5.1 K-Fold Cross Validation
- 5-Fold Stratified K-Fold
- Random Seed: 42
- Train/Val Split: 80/20

#### 5.2 하이퍼파라미터 튜닝
- XGBoost: Optuna 기반 최적화
  - 50 trials, Anti-overfitting 전략
  - Validation-LB gap 최소화

- DeepFM: Manual tuning
  - Embedding 차원 실험 (8, 10, 16, 32)
  - Hidden layer 크기 조정
  - Dropout/Weight Decay 튜닝

#### 5.3 과적합 방지
- XGBoost:
  - 강한 정규화 (L1, L2)
  - Subsample=0.88
  - Early stopping

- DeepFM:
  - Dropout 0.25
  - Weight Decay 2e-5
  - Early stopping (patience=7)

### 6. 앙상블 전략

#### 6.1 모델 선택
- XGBoost v6: Public LB 검증
- DeepFM v2.2: 높은 OOF (0.3507)
- DeepFM v3 제외: OOF가 v2.2보다 낮음

#### 6.2 앙상블 방법
- Weighted Average: 65% XGBoost + 35% DeepFM
- 가중치 선택 이유:
  * XGBoost의 안정성 중심
  * DeepFM으로 다양성 확보
  * 여러 비율 실험 후 최적 선택

#### 6.3 앙상블 효과
- Tree-based + Neural Network 상호보완
- 각 모델의 다른 에러 패턴 활용

### 7. 실험 과정 및 개선 이력

| 버전 | 모델 | 주요 변경 | OOF/LB | 비고 |
|------|------|----------|--------|------|
| v1 | DeepFM | 기본 구현 | - | 초기 버전 |
| v2 | DeepFM | Embedding 최적화 | 0.3507 | inventory_id 10D |
| v3 | DeepFM | Self-Attention 추가 | 0.3496 | 성능 저하 |
| v4 | XGBoost | Optuna 튜닝 | [Score] | 과적합 발견 |
| v6 | XGBoost | Anti-overfitting | [Score] | **최종 선택** |
| Final | Ensemble | XGB v6 + DeepFM v2.2 | [Score] | **제출** |

### 8. 실패한 시도 (Lessons Learned)

#### 8.1 DeepFM v3 (Self-Attention)
- **시도**: Multi-head Attention 추가
- **결과**: OOF 0.3496 (v2.2의 0.3507보다 낮음)
- **교훈**: 복잡한 모델이 항상 좋은 것은 아님

#### 8.2 xDeepFM (CIN)
- **시도**: Compressed Interaction Network
- **결과**: OOF ~0.346 (목표 미달)
- **교훈**: 데이터 특성에 맞는 모델 선택 중요

#### 8.3 3-Model Ensemble
- **시도**: XGB + DeepFM v2.2 + v3
- **결과**: 2-model 앙상블과 유사
- **교훈**: 모델 다양성보다 품질이 중요

### 9. 재현성

#### 9.1 Random Seed
- 모든 라이브러리에서 Seed 42 고정
- Python, NumPy, PyTorch, XGBoost

#### 9.2 환경
- Python 3.11.13
- PyTorch 2.0.1
- XGBoost 2.0.3
- Anaconda 환경

#### 9.3 검증
- inference.py 실행으로 Private Score 재현 가능
- 부동소수점 오차 < 1e-6

### 10. 결론 및 향후 개선 방향

#### 10.1 성과
- [x] Private Score: [Your Score]
- [x] 안정적인 앙상블 전략
- [x] 완전한 재현성 확보

#### 10.2 강점
- Tree-based + Neural Network 조합
- 체계적인 K-Fold 검증
- 과적합 방지 전략

#### 10.3 향후 개선 방향
- 추가 피처 엔지니어링
- Transformer 기반 모델 실험
- Stacking 앙상블 시도

---

## 📊 부록

### A. 하이퍼파라미터 상세
- XGBoost 전체 설정
- DeepFM 전체 설정

### B. 학습 곡선
- Loss/Score 그래프
- Learning Rate 스케줄

### C. Feature Importance
- Top 20 중요 피처
- Embedding 시각화

---

**작성자**: [Your Name]
**이메일**: [your.email@example.com]
**제출일**: 2025년 10월 17일
