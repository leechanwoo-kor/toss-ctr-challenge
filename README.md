# Toss CTR Challenge - 본선 제출용

XGBoost + DeepFM 2-Model Ensemble을 사용한 CTR 예측 솔루션입니다.

## 프로젝트 개요

- **대회**: Toss CTR Challenge
- **목표**: 클릭률(CTR) 예측
- **접근**: XGBoost와 DeepFM의 앙상블
- **성능**: Public LB 0.348+ (예상)

## 주요 특징

### 1. XGBoost 모델
- 5-Fold Cross Validation
- Optuna를 통한 하이퍼파라미터 최적화
- 풍부한 피처 엔지니어링

### 2. DeepFM 모델
- Dynamic Embedding 기법 적용
- 5-Fold Cross Validation
- GPU 가속 지원

### 3. 앙상블 전략
- XGBoost: 70% 가중치
- DeepFM: 30% 가중치
- Weighted Average 앙상블

## 프로젝트 구조

```
toss-ctr-challenge/
├── inference.py                    # 본선 제출용 추론 스크립트
├── requirements_frozen.txt         # 라이브러리 버전 고정
├── environment.yml                 # Conda 환경 설정
│
├── src/                            # 소스 코드
│   ├── deepfm/                     # DeepFM 관련 코드
│   ├── xgboost/                    # XGBoost 관련 코드
│   ├── models/                     # 모델 아키텍처
│   ├── utils/                      # 유틸리티
│   └── ensemble/                   # 앙상블
│
├── checkpoints/                    # 모델 가중치
│   ├── xgboost/
│   │   └── xgboost-kfold-*/
│   │       └── xgboost_fold_*.json
│   └── deepfm_kfold/
│       └── */
│           └── fold_*_model.pth
│
├── data/                           # 데이터
│   ├── train.parquet
│   ├── test.parquet
│   └── processed/
│
└── submissions/                    # 제출 파일
    └── ensemble/
```

## 환경 설정

### 필수 요구사항

- Python 3.8+
- CUDA 11.8+ (DeepFM 학습/추론용, 선택사항)

### 설치 방법

#### 1. Conda 환경 사용 (권장)

```bash
conda env create -f environment.yml
conda activate toss-ctr
```

#### 2. pip 사용

```bash
pip install -r requirements_frozen.txt
```

## 사용 방법

### Private Score 복원 (추론)

```bash
python inference.py
```

**출력**: `submissions/ensemble/inference_output_MMDD_HHMM.csv`

### 개별 모델 추론

#### XGBoost만 사용
```bash
python src/xgboost/xgboost_predict.py
```

#### DeepFM만 사용
```bash
python src/deepfm/deepfm_predict_kfold.py
```

### 모델 재학습 (선택사항)

#### 1. 데이터 전처리

```bash
# XGBoost
python src/xgboost/xgboost_prepare_data.py

# DeepFM
python src/deepfm/deepfm_prepare_data_new.py
```

#### 2. 모델 학습

```bash
# XGBoost
python src/xgboost/xgboost_train.py

# DeepFM
python src/deepfm/deepfm_train_kfold.py
```

## 모델 성능

### XGBoost
- Public LB: 0.348568
- 5-Fold CV 평균: 0.XXXX

### DeepFM v2.2
- Public LB: 0.346596
- 5-Fold CV 평균: 0.XXXX

### 앙상블 (70% XGB + 30% DeepFM)
- Public LB: 0.348+ (예상)

## 주요 기법

### 피처 엔지니어링
- Sequence 피처 처리
- EDA 기반 피처 생성
- Inventory 인기도, 친화도 피처
- 시간대/요일 피처

### 모델 최적화
- Optuna를 통한 하이퍼파라미터 튜닝
- 5-Fold Cross Validation
- Early Stopping
- Dynamic Embedding (DeepFM)

## 파일 설명

- **inference.py**: 본선 제출용 통합 추론 스크립트
- **MODEL_WEIGHTS.md**: 모델 가중치 파일 상세 설명
- **SUBMISSION_CHECKLIST.md**: 제출 체크리스트

## 주의사항

1. **경로**: 모든 경로는 상대 경로로 설정됨
2. **인코딩**: 모든 파일 UTF-8 인코딩
3. **GPU**: DeepFM은 GPU 사용 시 더 빠름 (CPU도 가능)
4. **메모리**: 추론 시 최소 16GB RAM 권장

## 제출 파일

1. 전체 개발 코드
2. 모델 가중치 파일 (10개)
3. 전처리 객체 파일 (5개)
4. requirements_frozen.txt
5. 모델 개발 보고서 (별도 제출)

## 라이선스

This project is for Toss CTR Challenge submission.

## 문의

본선 제출 관련 문의: dacon@dacon.io
