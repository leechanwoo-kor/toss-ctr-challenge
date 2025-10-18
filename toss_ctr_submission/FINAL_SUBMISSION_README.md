# 토스 CTR Challenge - 최종 제출 패키지

**제출일**: 2025년 10월 17일
**모델**: XGBoost v6 (65%) + DeepFM v2.2 (35%) 앙상블

---

## 📦 패키지 구성

```
toss_ctr_submission/
├── submissions/
│   └── ensemble_xgb_deepfm_65_35_1012_1728.csv  ⭐ 최종 제출 파일 (Private Score 재현)
├── checkpoints/
│   ├── xgboost/          # XGBoost v6 5-fold 모델 (43MB)
│   └── deepfm/           # DeepFM v2.2 5-fold 모델 (7MB)
├── train/
│   ├── xgboost_train.py  # XGBoost v6 학습 코드
│   └── deepfm_train.py   # DeepFM v2.2 학습 코드
├── src/                  # 모델 및 유틸리티 코드
├── data/
│   └── processed/deepfm_new/processor.pkl  # DeepFM 전처리기
├── report/
│   └── MODEL_DEVELOPMENT_REPORT.md  # 모델 개발 보고서
├── environment.yml       # Conda 환경
├── requirements.txt      # Python 패키지
└── README.md             # 일반 가이드
```

---

## 🎯 최종 제출 파일

**파일명**: `ensemble_xgb_deepfm_65_35_1012_1728.csv`
**위치**: `submissions/ensemble_xgb_deepfm_65_35_1012_1728.csv`
**생성일시**: 2025년 10월 12일 17:28
**행 수**: 1,527,298 (test.parquet의 모든 샘플)

### 생성 방법

1. **XGBoost v6 예측** (65% 가중치)
   - 5-Fold 모델 각각으로 test 데이터 예측
   - 5개 예측의 평균 계산
   - 체크포인트: `checkpoints/xgboost/xgboost_fold_{1-5}.json`

2. **DeepFM v2.2 예측** (35% 가중치)
   - 5-Fold 모델 각각으로 test 데이터 예측
   - 5개 예측의 평균 계산
   - 체크포인트: `checkpoints/deepfm/fold_{1-5}_model.pth`

3. **앙상블**
   ```python
   final_predictions = 0.65 * xgboost_predictions + 0.35 * deepfm_predictions
   ```

---

## 🔧 학습 재현 방법

### 환경 설정

```bash
# Conda 환경 생성
conda env create -f environment.yml
conda activate ctr

# 또는 pip 사용
pip install -r requirements.txt
```

### 데이터 준비

```
data/
├── train.parquet  # 원본 훈련 데이터 (10.7M rows)
└── test.parquet   # 원본 테스트 데이터 (1.5M rows)
```

### XGBoost v6 학습

```bash
python train/xgboost_train.py
```

**출력**:
- `checkpoints/xgboost/xgboost_fold_{1-5}.json` (5개 fold 모델)
- Out-of-Fold Score: 0.353X (예상)

**하이퍼파라미터** (src/utils/config.py - XGBOOST_CONFIG_V6):
- max_depth: 9
- learning_rate: 0.0392
- n_estimators: 245
- min_child_weight: 265
- reg_lambda: 0.32 (L2 정규화)
- gamma: 7.6
- scale_pos_weight: 51.6

### DeepFM v2.2 학습

```bash
python train/deepfm_train.py
```

**출력**:
- `checkpoints/deepfm/fold_{1-5}_model.pth` (5개 fold 모델)
- Out-of-Fold Score: 0.3507

**아키텍처**:
- Embedding: inventory_id 10D, 기타 16D
- Hidden layers: [256, 128]
- Dropout: 0.25
- Loss: BCEWithLogitsLoss (pos_weight=51.6)

---

## 📊 재현성

### Random Seed 고정

모든 코드에서 Random Seed 42 고정:

```python
# Python, NumPy
random.seed(42)
np.random.seed(42)

# PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# XGBoost
params['random_state'] = 42

# Scikit-learn
StratifiedKFold(n_splits=5, random_state=42)
```

### 학습 재현성

- ✅ **XGBoost**: 동일한 seed + 데이터로 완전히 재현 가능
- ✅ **DeepFM**: 동일한 seed + 데이터로 재현 가능 (부동소수점 오차 < 1e-6)

### 추론 재현성 (완전 재현 가능!)

**현재 상태**:
- ✅ **실제 사용한 추론 스크립트 3개 포함**
- ✅ 최종 제출 파일(`ensemble_xgb_deepfm_65_35_1012_1728.csv`) 제공됨
- ✅ XGBoost/DeepFM processor 포함 (data/processed/)

**추론 워크플로우** (3단계):

#### Step 1: XGBoost 예측 생성
```bash
python src/xgboost/xgboost_predict.py
```
- **Processor 사용**: `data/processed/xgboost/processor.pkl`
- **모델 로드**: `checkpoints/xgboost/xgboost_fold_{1-5}.json`
- **출력**: `submissions/xgboost/xgboost_kfold_MMDD_HHMM.csv`
- **K-Fold 앙상블**: 5개 fold 모델의 평균 예측

#### Step 2: DeepFM 예측 생성
```bash
python src/deepfm/deepfm_predict_kfold.py
```
- **Processor 사용**: `data/processed/deepfm_new/processor.pkl`
- **모델 로드**: `checkpoints/deepfm/fold_{1-5}_model.pth`
- **출력**: `submissions/deepfm/deepfm_kfold_ensemble_YYYYMMDD_HHMMSS.csv`
- **K-Fold 앙상블**: 5개 fold 모델의 평균 예측

#### Step 3: 2-Model 앙상블
```bash
python src/ensemble/two_model_ensemble.py
```
- **입력**: XGBoost CSV + DeepFM CSV
- **가중치**: 65% XGBoost + 35% DeepFM
- **출력**: `submissions/ensemble/ensemble_xgb_deepfm_65_35_MMDD_HHMM.csv`
- **여러 가중치 조합 자동 생성** (80:20, 75:25, 70:30, 65:35, 60:40, 50:50)

**Private Score 재현 방법**:
1. 위 3단계 순차 실행
2. 또는 제공된 최종 제출 파일 직접 사용

---

## 📈 성능 지표

### Out-of-Fold (OOF) Scores

| 모델 | OOF Score | Public LB (예상) | 비고 |
|------|-----------|------------------|------|
| XGBoost v6 | 0.353X | 0.351X | Anti-overfitting 최적화 |
| DeepFM v2.2 | 0.3507 | 0.3466 | Embedding 최적화 (inventory_id 10D) |
| **Ensemble (65:35)** | **0.355X** | **0.350-0.352** | **최종 제출** ⭐ |

### K-Fold Cross Validation

- **5-Fold Stratified CV**
- Stratified sampling으로 클래스 비율 유지 (clicked: 1.94%)
- 각 fold: Train 80% (8.6M), Validation 20% (2.1M)

---

## 🔍 모델 상세 정보

### XGBoost v6 특징

**Anti-Overfitting 전략**:
- v4 (CV 0.355, LB 0.349, gap -0.006) → 과적합 발견
- v5: 극단적 정규화 (n_estimators 210)
- **v6: 최적 균형점** (n_estimators 245, reg_lambda 0.32)

**피처 엔지니어링**:
- Categorical: Label Encoding
- Numerical: QuantileTransformer (정규분포 변환)
- EDA Features:
  - inventory_popularity_rank
  - gender_age_combo
  - age_inventory_affinity
  - gender_inventory_affinity

### DeepFM v2.2 특징

**아키텍처**:
```
Input (Categorical + Numerical)
    ↓
Embedding (inventory_id: 10D, others: 16D)
    ↓
├─── FM Component ────┬─── Deep Component ───┐
│ (1st + 2nd order)   │   ([256, 128] DNN)   │
└─────────────────────┴──────────────────────┘
                     ↓
                  Sigmoid
                     ↓
                  Output
```

**최적화**:
- inventory_id: 45K 카디널리티 → 10D embedding (파라미터 37% 감소)
- 기타 categorical: 16D embedding
- CosineAnnealingLR 스케줄러
- Dropout 0.25 + Weight Decay 2e-5

---

## 📝 주요 파일 설명

### 학습 코드

- **`train/xgboost_train.py`**: XGBoost v6 5-Fold 학습
- **`train/deepfm_train.py`**: DeepFM v2.2 5-Fold 학습

### 모델 정의

- **`src/models/deepfm_optimized.py`**: DeepFM v2.2 아키텍처
- **`src/deepfm/deepfm_processor.py`**: DeepFM 전처리기

### 설정

- **`src/utils/config.py`**: 모든 하이퍼파라미터 정의
- **`src/utils/metrics.py`**: Competition Score 계산 (0.9×AUC + 0.1×AP)

---

## 🚀 제출 절차

### 1. 패키지 압축

```bash
cd D:\ctr
zip -r toss_ctr_submission.zip toss_ctr_submission/
# 또는
7z a toss_ctr_submission.7z toss_ctr_submission/
```

### 2. 제출 파일 확인

필수 포함 파일:
- ✅ `submissions/ensemble_xgb_deepfm_65_35_1012_1728.csv` (Private Score 재현)
- ✅ `checkpoints/xgboost/` (43MB, 5 모델)
- ✅ `checkpoints/deepfm/` (7MB, 5 모델)
- ✅ `train/*.py` (학습 코드)
- ✅ `src/` (전체 소스 코드)
- ✅ `report/MODEL_DEVELOPMENT_REPORT.md` (개발 보고서)
- ✅ `environment.yml`, `requirements.txt`
- ✅ `README.md`, `SUBMISSION_GUIDE.md`

### 3. 이메일 제출

- **수신**: dacon@dacon.io
- **제목**: [토스 NEXT ML CHALLENGE] 최종 제출 - [팀명/이름]
- **첨부**: toss_ctr_submission.zip (또는 .7z)
- **본문**:
  ```
  안녕하세요,

  토스 NEXT ML CHALLENGE 본선 제출물입니다.

  - 최종 제출 파일: ensemble_xgb_deepfm_65_35_1012_1728.csv
  - 모델: XGBoost v6 (65%) + DeepFM v2.2 (35%)
  - Out-of-Fold Score: 0.355X

  감사합니다.
  ```

---

## 📖 참고 문서

- **`README.md`**: 일반 가이드
- **`SUBMISSION_GUIDE.md`**: 제출 가이드 (상세)
- **`report/MODEL_DEVELOPMENT_REPORT.md`**: 모델 개발 보고서 (50+ 페이지)
  - EDA, 피처 엔지니어링, 모델 아키텍처
  - 실험 과정, 실패 사례, Lessons Learned
  - 재현성, 향후 개선 방향

---

## ⚠️ 알려진 제한사항

### 1. 학습 데이터 미포함

- **이유**: 파일 크기 제약 (train.parquet 8.2GB, test.parquet 1.2GB)
- **대안**: 대회 페이지에서 다운로드 후 `data/` 폴더에 배치
- **필수 파일**:
  - `data/train.parquet` - 훈련 데이터
  - `data/test.parquet` - 테스트 데이터

### 2. XGBoost/DeepFM Processor 포함

- ✅ **DeepFM Processor**: `data/processed/deepfm_new/processor.pkl` (947KB)
- ✅ **XGBoost Processor**: `data/processed/xgboost/processor.pkl` (7.4KB)
- ✅ **XGBoost 추가 파일**: `feature_names.pkl` (1.5KB), `metadata.pkl` (571B)
- **완전한 재현**: 모든 processor 포함으로 추론 완전 재현 가능!

### 3. 중간 예측 파일 미포함

- **미포함 파일**:
  - `submissions/xgboost/xgboost_kfold_1012_1727.csv`
  - `submissions/deepfm/deepfm_kfold_ensemble_20251011_000856.csv`
- **이유**: 파일 크기 제약 (각 1.2GB+)
- **대안**: 위 추론 워크플로우로 재생성 가능

---

## 🎓 Lessons Learned

### 성공 요인

1. **체계적인 과적합 방지**: XGBoost v4 → v6 진화 과정
2. **Embedding 최적화**: DeepFM inventory_id 10D로 파라미터 효율화
3. **상호보완적 앙상블**: Tree-based + Neural Network 조합

### 개선 필요 사항

1. **Processor 저장**: 피처 엔지니어링 로직을 pickle로 저장
2. **추론 파이프라인**: end-to-end inference 구현
3. **문서화**: 초기부터 재현성을 고려한 코드 작성

---

## 📞 문의

제출 패키지 관련 문의사항은 대회 운영진에게 문의 바랍니다.

---

**작성일**: 2025년 10월 17일
**작성자**: [Your Name]
**대회**: 토스 NEXT ML CHALLENGE - 광고 클릭 예측(CTR) 모델 개발
