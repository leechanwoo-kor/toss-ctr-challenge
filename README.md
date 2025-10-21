# Toss CTR Prediction Challenge - 본선 제출 산출물

**대회명**: Toss CTR 예측 알고리즘 경진대회
**제출일**: 2025년 10월 12일
**팀명/이름**: [팀명 입력]

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [환경 설정](#환경-설정)
3. [재현 방법](#재현-방법)
4. [모델 구조](#모델-구조)
5. [디렉토리 구조](#디렉토리-구조)
6. [결과](#결과)

---

## 🎯 프로젝트 개요

본 프로젝트는 사용자의 광고 클릭 여부(CTR)를 예측하는 이진 분류 모델입니다.

### 최종 제출 구성
- **XGBoost K-Fold 모델** (5-Fold Cross Validation)
- **DeepFM K-Fold 모델** (5-Fold Cross Validation)
- **앙상블**: XGBoost + DeepFM 가중 평균

### 성능
- **Private Leaderboard Score**: [점수 입력]
- **Public Leaderboard Score**: [점수 입력]

---

## 🔧 환경 설정

### 필수 요구사항
- Python 3.10+
- CUDA 11.8+ (GPU 사용 시)
- Conda 또는 pip

### 설치 방법

#### 1. Conda 환경 생성 (권장)
```bash
conda create -n ctr python=3.10
conda activate ctr
```

#### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

#### 3. PyTorch 설치
```bash
# GPU 사용 (CUDA 11.8 권장)
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 또는 CUDA 12.1
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121

# CPU only (GPU 없는 경우)
pip install torch==2.7.1
```

**참고**: `requirements.txt`의 `torch==2.7.1`은 CPU 버전입니다. GPU를 사용하려면 위 명령어로 재설치하세요.

### 주요 패키지 버전
- pandas==2.3.2
- numpy==2.3.3
- scikit-learn==1.7.2
- xgboost==3.0.5
- torch==2.7.1
- pyarrow==21.0.0

---

## 🚀 재현 방법

### 빠른 시작 (추론만 실행)

**inference.py**를 실행하면 10월 12일 최종 제출을 완벽하게 재현할 수 있습니다.

```bash
# Conda 환경 활성화
conda activate ctr

# 추론 실행
python inference.py
```

**실행 시간**: GPU 사용 시 약 3-5분

**생성되는 파일**:
```
submissions/reproduction/
├── xgboost_reproduction.csv          # XGBoost 예측
├── deepfm_reproduction.csv           # DeepFM 예측
├── ensemble_xgb80_deepfm20_reproduction.csv
├── ensemble_xgb75_deepfm25_reproduction.csv
├── ensemble_xgb70_deepfm30_reproduction.csv
├── ensemble_xgb65_deepfm35_reproduction.csv
├── ensemble_xgb60_deepfm40_reproduction.csv
└── ensemble_xgb50_deepfm50_reproduction.csv
```

### 재현 검증

생성된 파일이 원본 제출과 동일한지 확인:

```bash
python -c "
import pandas as pd
import numpy as np

# DeepFM 비교
original = pd.read_csv('submissions/deepfm/deepfm_kfold_ensemble_20251011_000856.csv')
reproduced = pd.read_csv('submissions/reproduction/deepfm_reproduction.csv')
diff = np.abs(original['clicked'] - reproduced['clicked'])
print(f'DeepFM 차이: {diff.mean():.10f} (0이면 완벽한 재현)')

# XGBoost 비교
original = pd.read_csv('submissions/xgboost/xgboost_kfold_1012_1727.csv')
reproduced = pd.read_csv('submissions/reproduction/xgboost_reproduction.csv')
diff = np.abs(original['clicked'] - reproduced['clicked'])
print(f'XGBoost 차이: {diff.mean():.10f} (0이면 완벽한 재현)')
"
```

**예상 결과**: 두 모델 모두 차이 0.0000000000 (완벽한 재현)

---

## 🏗️ 모델 구조

### 1. XGBoost K-Fold 모델

**아키텍처**:
- Gradient Boosting Decision Tree
- 5-Fold Stratified Cross Validation
- Out-of-Fold 예측 평균

**주요 하이퍼파라미터**:
- Learning Rate: 0.01
- Max Depth: 7
- Subsample: 0.8
- Colsample_bytree: 0.8
- Tree Method: hist (GPU 가속)

**피처 엔지니어링**:
- 범주형 피처: gender, age_group, inventory_id, hour, day_of_week
- 수치형 피처: feat_a, feat_b, l_feat 계열 등 118개
- EDA 기반 피처: inventory popularity, frequency, 시간 cyclical encoding 등

**체크포인트**:
- `checkpoints/xgboost/xgboost-kfold-1012_1718/`
- 5개 모델 파일: `xgboost_fold_1.json` ~ `xgboost_fold_5.json`

### 2. DeepFM K-Fold 모델

**아키텍처**:
- **FM (Factorization Machine)**: 2차 피처 상호작용 학습
- **Deep Neural Network**: 고차원 비선형 패턴 학습
- Dynamic Embedding: 카디널리티에 따라 embedding 크기 조정 (4~50차원)

**모델 구조**:
```
Input (Categorical + Numerical Features)
    │
    ├─→ FM Component (2nd-order interactions)
    │
    └─→ Deep Component
          ├─ Embedding Layer (dynamic size)
          ├─ Hidden Layer 1: 256 units + Dropout(0.25)
          ├─ Hidden Layer 2: 128 units + Dropout(0.25)
          └─ Output Layer
    │
    └─→ Combine (FM + Deep) → Sigmoid → CTR Prediction
```

**주요 하이퍼파라미터**:
- Embedding 크기: `6 * cardinality^0.25` (최소 4, 최대 50)
- Hidden Units: [256, 128]
- Dropout: 0.25
- Learning Rate: 0.001
- Optimizer: AdamW (weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Early Stopping: 5 epochs
- Batch Size: 1024

**피처 처리**:
- 범주형: gender, age_group, inventory_id, hour, day_of_week (5개)
- 수치형: 119개 (스케일링: QuantileTransformer)
- 결측치 인디케이터, cyclical encoding 등 포함

**체크포인트**:
- `checkpoints/deepfm_kfold/deepfm-dynamic-embedding-kfold-1010_1642/`
- 5개 모델 파일: `fold_1_model.pth` ~ `fold_5_model.pth`
- OOF Score: 0.351232
- CV Score: 0.351286 ± 0.001013

### 3. 앙상블 전략

**방법**: Weighted Average
**가중치 조합**: 6가지 비율 테스트
- XGBoost 80% + DeepFM 20%
- XGBoost 75% + DeepFM 25%
- XGBoost 70% + DeepFM 30%
- XGBoost 65% + DeepFM 35%
- XGBoost 60% + DeepFM 40%
- XGBoost 50% + DeepFM 50%

**최종 제출**: XGBoost 65% + DeepFM 35%

---

## 📁 디렉토리 구조

```
.
├── data/
│   ├── train.parquet                # 원본 학습 데이터
│   ├── test.parquet                 # 원본 테스트 데이터
│   └── processed/
│       ├── deepfm_new/
│       │   ├── processor.pkl        # DeepFM 전처리 프로세서
│       │   ├── metadata.pkl
│       │   ├── train_dataset.pt
│       │   └── val_dataset.pt
│       └── xgboost/
│           ├── processor.pkl        # XGBoost 전처리 프로세서
│           ├── feature_names.pkl
│           ├── train_processed.parquet
│           └── val_processed.parquet
│
├── checkpoints/
│   ├── deepfm_kfold/
│   │   └── deepfm-dynamic-embedding-kfold-1010_1642/
│   │       ├── fold_1_model.pth
│   │       ├── fold_2_model.pth
│   │       ├── fold_3_model.pth
│   │       ├── fold_4_model.pth
│   │       ├── fold_5_model.pth
│   │       ├── kfold_results.pkl
│   │       └── oof_predictions.npy
│   └── xgboost/
│       └── xgboost-kfold-1019_0024/
│           ├── xgboost_fold_1.json
│           ├── xgboost_fold_2.json
│           ├── xgboost_fold_3.json
│           ├── xgboost_fold_4.json
│           ├── xgboost_fold_5.json
│           ├── kfold_results.pkl
│           └── oof_predictions.npy
│
├── src/
│   ├── deepfm/
│   │   ├── deepfm_prepare_data_new.py   # DeepFM 데이터 전처리
│   │   ├── deepfm_train_kfold.py        # DeepFM K-Fold 학습
│   │   ├── deepfm_predict_kfold.py      # DeepFM K-Fold 예측
│   │   └── deepfm_processor.py          # DeepFM 프로세서
│   ├── xgboost/
│   │   ├── xgboost_prepare_data.py      # XGBoost 데이터 전처리
│   │   ├── xgboost_train.py             # XGBoost K-Fold 학습
│   │   └── xgboost_predict.py           # XGBoost K-Fold 예측
│   ├── models/
│   │   ├── deepfm_dynamic_embedding.py  # DeepFM 모델 정의
│   │   └── xgboost_feature_processor.py # XGBoost 피처 프로세서
│   ├── ensemble/
│   │   └── two_model_ensemble.py        # 2-모델 앙상블
│   └── utils/
│       ├── metrics.py                   # 평가 메트릭
│       └── config.py                    # 설정 파일
│
├── submissions/
│   ├── xgboost/
│   │   └── xgboost_kfold_1012_1727.csv      # 10월 12일 제출 XGBoost
│   ├── deepfm/
│   │   └── deepfm_kfold_ensemble_20251011_000856.csv  # 10월 12일 제출 DeepFM
│   ├── ensemble/
│   │   └── ensemble_xgb_deepfm_*_1012_1728.csv  # 10월 12일 제출 앙상블 (6개)
│   └── reproduction/
│       └── [inference.py로 생성된 재현 파일들]
│
├── inference.py              # ⭐ 재현용 추론 스크립트
├── requirements.txt          # 패키지 의존성
└── README.md                 # 본 문서
```

---

## 📊 결과

### K-Fold Cross Validation 결과

#### DeepFM (1010_1642 체크포인트)
| Fold | Competition Score |
|------|-------------------|
| 1    | 0.3521            |
| 2    | 0.3507            |
| 3    | 0.3518            |
| 4    | 0.3502            |
| 5    | 0.3515            |
| **Mean** | **0.3513 ± 0.0010** |
| **OOF**  | **0.3512** |

### 재현성 검증

| 모델 | 원본 예측 | 재현 예측 | 절대 차이 (평균) | 일치율 |
|------|-----------|-----------|----------|--------|
| DeepFM | deepfm_kfold_ensemble_20251011_000856.csv | deepfm_reproduction.csv | **0.0000000000** | **100%** |
| XGBoost | xgboost_kfold_1012_1727.csv | xgboost_reproduction.csv | **0.0000000000** | **100%** |
| Ensemble 65_35 | ensemble_xgb_deepfm_65_35_1012_1728.csv | ensemble_xgb65_deepfm35_reproduction.csv | **0.0000000041** | **실질적으로 동일** |

**결론**: 모든 예측이 완벽하게 재현됩니다. 앙상블에서 미세한 부동소수점 오차(~10^-9)가 있으나 실질적으로 동일합니다.

---

## 🔍 주요 특징

### 1. 완벽한 재현성
- ✅ Seed 고정 (모든 랜덤 요소 제어)
- ✅ 체크포인트 명시적 지정
- ✅ 동일한 데이터 전처리 프로세서 사용
- ✅ 100% 동일한 예측 결과

### 2. 견고한 검증
- ✅ 5-Fold Stratified Cross Validation
- ✅ Out-of-Fold 예측으로 과적합 방지
- ✅ 앙상블로 일반화 성능 향상

### 3. 효율적인 구현
- ✅ GPU 가속 (CUDA)
- ✅ 메모리 효율적인 배치 처리
- ✅ 빠른 추론 (3-5분)

---

## 📝 추가 정보

### 학습 방법 (참고용)

본 재현 스크립트는 **추론(inference)만** 수행합니다.
전체 학습 파이프라인은 다음과 같습니다:

#### 1. 데이터 전처리
```bash
# DeepFM
python src/deepfm/deepfm_prepare_data_new.py

# XGBoost
python src/xgboost/xgboost_prepare_data.py
```

#### 2. 모델 학습
```bash
# DeepFM (GPU 권장, ~1-2시간)
python src/deepfm/deepfm_train_kfold.py

# XGBoost (GPU 권장, ~30분)
python src/xgboost/xgboost_train.py
```

#### 3. 예측 생성
```bash
# DeepFM
python src/deepfm/deepfm_predict_kfold.py

# XGBoost
python src/xgboost/xgboost_predict.py
```

#### 4. 앙상블
```bash
python src/ensemble/two_model_ensemble.py
```

**참고**: 학습은 시간이 오래 걸리므로, 제공된 체크포인트를 사용하여 `inference.py`로 바로 재현하는 것을 권장합니다.

---

## ⚠️ 알려진 이슈 및 제한사항

1. **Windows 환경 이모지 출력 문제**
   - 일부 스크립트에서 이모지 사용으로 인한 인코딩 에러 가능
   - 해결: `inference.py`는 이모지 없이 작성되어 문제 없음

2. **메모리 요구사항**
   - DeepFM 학습: 최소 16GB RAM 권장
   - 추론: 8GB RAM으로 충분

3. **GPU 권장 (CPU도 지원)**
   - inference.py는 GPU/CPU 자동 감지 및 실행
   - GPU 사용 시: 3-5분
   - CPU 사용 시: 30-40분 (약 10배 느림)
   - CUDA 없는 환경에서도 정상 작동

---

## 📧 문의

**이름**: [이름]
**이메일**: [이메일]
**GitHub**: [GitHub URL]

---

## 📄 라이센스

본 프로젝트는 Toss CTR 예측 알고리즘 경진대회의 일환으로 작성되었습니다.

---

**마지막 업데이트**: 2025년 10월 21일
