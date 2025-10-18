# Toss CTR Challenge - 본선 제출물

**팀명**: [Your Team Name]
**참가자**: [Your Name]
**제출일**: 2025-10-12
**Private Score**: [Your Score]

---

## 📋 목차

1. [개요](#개요)
2. [환경 설정](#환경-설정)
3. [파일 구조](#파일-구조)
4. [실행 방법](#실행-방법)
5. [모델 설명](#모델-설명)
6. [재현성](#재현성)

---

## 개요

### 최종 제출 모델
- **앙상블 구성**: XGBoost v6 (65%) + DeepFM v2.2 (35%)
- **최종 제출 파일**: `ensemble_xgb_deepfm_65_35_1012_1728.csv`
- **Private Score**: [Your Private Score]

### 핵심 특징
- Tree-based (XGBoost) + Neural Network (DeepFM) 앙상블
- K-Fold Cross Validation (5-Fold) 사용
- 완전한 재현 가능성 (Random Seed 고정)

---

## 환경 설정

### 시스템 요구사항
- **OS**: Windows 10/11 (개발 환경 기준)
- **Python**: 3.11.13
- **CUDA**: 11.8 (GPU 사용 시)
- **RAM**: 최소 16GB 권장
- **Storage**: 최소 10GB 여유 공간

### Anaconda 환경 생성

```bash
# 1. Conda 환경 생성 (environment.yml 사용)
conda env create -f environment.yml

# 2. 환경 활성화
conda activate ctr

# 또는 pip로 설치 (requirements.txt 사용)
pip install -r requirements.txt
```

### 주요 라이브러리 버전
- `pandas==2.0.3`
- `numpy==1.24.3`
- `torch==2.0.1`
- `xgboost==2.0.3`
- `scikit-learn==1.3.2`
- `catboost==1.2.8`
- `lightgbm==4.1.0`

전체 라이브러리 목록은 `environment.yml` 또는 `requirements.txt` 참조

---

## 파일 구조

```
toss-ctr-submission/
│
├── README_SUBMISSION.md          # 본 파일
├── environment.yml                # Conda 환경 설정
├── requirements.txt               # Pip 패키지 목록
├── inference.py                   # 최종 추론 스크립트 ⭐
│
├── data/
│   ├── train.csv                  # 학습 데이터 (제공)
│   ├── test.csv                   # 테스트 데이터 (제공)
│   └── processed/                 # 전처리된 데이터
│       └── deepfm_new/
│           ├── processor.pkl      # DeepFM 전처리 프로세서
│           └── train_dataset.pt   # 전처리된 학습 데이터
│
├── checkpoints/                   # 모델 가중치 ⭐
│   ├── xgboost/
│   │   └── xgboost_v6_model.pkl  # XGBoost v6 모델
│   └── deepfm_kfold/
│       └── deepfm-kfold-v2.2/    # DeepFM v2.2 5-Fold 모델
│           ├── fold_1_model.pth
│           ├── fold_2_model.pth
│           ├── fold_3_model.pth
│           ├── fold_4_model.pth
│           └── fold_5_model.pth
│
├── src/                           # 소스 코드
│   ├── config.py                  # 전역 설정
│   ├── utils/
│   │   ├── config.py             # 모델별 하이퍼파라미터
│   │   └── metrics.py            # 평가 메트릭
│   ├── models/
│   │   ├── deepfm_optimized.py   # DeepFM 모델 정의
│   │   └── feature_processor.py  # 피처 전처리
│   ├── xgboost/
│   │   ├── xgboost_train.py      # XGBoost 학습 코드
│   │   └── xgboost_predict.py    # XGBoost 추론 코드
│   ├── deepfm/
│   │   ├── deepfm_processor.py   # DeepFM 전처리
│   │   ├── deepfm_train_kfold.py # DeepFM 학습 코드
│   │   └── deepfm_predict_kfold.py # DeepFM 추론 코드
│   └── ensemble/
│       └── two_model_ensemble.py # 앙상블 코드
│
├── submissions/                   # 제출 파일
│   └── final/
│       └── ensemble_xgb_deepfm_65_35_final.csv  # 최종 제출 결과
│
└── reports/                       # 보고서
    └── model_development_report.pdf  # 모델 개발 보고서 ⭐
```

---

## 실행 방법

### 1. 데이터 준비
```bash
# data/ 디렉토리에 train.csv, test.csv 배치
# (대회에서 제공된 데이터)
```

### 2. Private Score 재현 (추론만 실행)

```bash
# inference.py 실행으로 최종 제출 파일 재현
python inference.py
```

**출력**:
- `submissions/final/ensemble_xgb_deepfm_65_35_final.csv`
- 이 파일이 Private Score를 재현합니다

**실행 시간**: 약 10-15분 (GPU 사용 시)

### 3. 전체 학습 과정 재현 (선택사항)

학습부터 전체 과정을 재현하려면:

```bash
# 1. XGBoost 학습
python src/xgboost/xgboost_train.py

# 2. DeepFM 데이터 전처리
python src/deepfm/deepfm_prepare_data_new.py

# 3. DeepFM 학습
python src/deepfm/deepfm_train_kfold.py

# 4. 앙상블 생성
python src/ensemble/two_model_ensemble.py
```

**전체 학습 시간**: 약 3-4시간 (GPU 사용 시)

---

## 모델 설명

### 1. XGBoost v6

**아키텍처**:
- Gradient Boosting Decision Tree
- K-Fold Cross Validation (5-Fold)

**하이퍼파라미터**:
```python
{
    "max_depth": 9,
    "learning_rate": 0.039,
    "n_estimators": 280,
    "subsample": 0.88,
    "colsample_bytree": 0.72,
    "reg_alpha": 0.27,
    "reg_lambda": 0.40,
    "scale_pos_weight": 51.6,
}
```

**특징**:
- Tree-based 모델로 피처 간 interaction 학습
- Overfitting 방지를 위한 강력한 정규화
- Class imbalance 처리 (scale_pos_weight)

### 2. DeepFM v2.2

**아키텍처**:
- FM (Factorization Machine) + Deep Neural Network
- Embedding 차원 최적화: inventory_id (10D), 기타 (16D)
- K-Fold Cross Validation (5-Fold)

**하이퍼파라미터**:
```python
{
    "embedding_dim": 16,
    "hidden_dims": [256, 128],
    "dropout": 0.25,
    "learning_rate": 0.001,
    "batch_size": 1024,
    "epochs": 40,
    "weight_decay": 2e-5,
    "pos_weight": 51.6,
}
```

**특징**:
- FM Part: 2차 피처 interaction 명시적 학습
- Deep Part: 고차원 비선형 패턴 학습
- Embedding 최적화로 파라미터 효율성 향상

### 3. 앙상블 전략

**가중치**:
- XGBoost: 65%
- DeepFM: 35%

**선택 이유**:
- XGBoost: 안정적인 Public LB 성능
- DeepFM: Neural network의 비선형 패턴 보완
- 65:35 비율로 XGBoost의 안정성 중심, DeepFM으로 다양성 확보

---

## 재현성

### Random Seed 고정
모든 코드에서 Random Seed 42 사용:
- Python random
- NumPy
- PyTorch
- XGBoost

### 재현성 검증 방법

1. **추론 재현**:
```bash
python inference.py
```

2. **결과 비교**:
```bash
# 생성된 파일과 원본 제출 파일 비교
python -c "
import pandas as pd
original = pd.read_csv('submissions/ensemble_xgb_deepfm_65_35_1012_1728.csv')
reproduced = pd.read_csv('submissions/final/ensemble_xgb_deepfm_65_35_final.csv')
diff = (original['clicked'] - reproduced['clicked']).abs()
print(f'Max difference: {diff.max():.10f}')
print(f'Mean difference: {diff.mean():.10f}')
"
```

**예상 결과**:
- Max difference < 1e-6 (부동소수점 오차 범위)
- 완전 재현 가능

### 주의사항

1. **GPU vs CPU**: GPU/CPU에 따라 부동소수점 연산 미세 차이 가능
2. **PyTorch 버전**: PyTorch 2.0.1 사용 권장
3. **XGBoost 버전**: XGBoost 2.0.3 사용 필수

---

## 성능 요약

### Private 리더보드
- **Private Score**: [Your Private Score]
- **Rank**: [Your Rank]

### 교차 검증 성능
- **XGBoost OOF**: [XGBoost Score]
- **DeepFM OOF**: 0.3507
- **앙상블 예상**: [Ensemble Score]

---

## 문의

질문이나 문제 발생 시 연락처:
- **Email**: [your.email@example.com]
- **GitHub**: [your-github-username]

---

**제출일**: 2025년 10월 17일
**대회**: 토스 NEXT ML CHALLENGE - 광고 클릭 예측(CTR) 모델 개발
