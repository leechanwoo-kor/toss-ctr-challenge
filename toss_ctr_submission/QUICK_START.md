# 토스 CTR Challenge - Quick Start Guide

**최종 제출 파일**: `submissions/ensemble_xgb_deepfm_65_35_1012_1728.csv`

---

## 🚀 빠른 시작 (추론만)

이미 학습된 모델로 추론 재현:

### Step 1: 환경 설정
```bash
conda env create -f environment.yml
conda activate ctr
```

### Step 2: 데이터 준비
```
data/
├── train.parquet  # 대회에서 다운로드 (8.2GB)
└── test.parquet   # 대회에서 다운로드 (1.2GB)
```

### Step 3: 추론 실행 (3단계)

```bash
# 1. XGBoost 예측 생성 (5-10분)
python src/xgboost/xgboost_predict.py
# 출력: submissions/xgboost/xgboost_kfold_MMDD_HHMM.csv

# 2. DeepFM 예측 생성 (10-15분, GPU 권장)
python src/deepfm/deepfm_predict_kfold.py
# 출력: submissions/deepfm/deepfm_kfold_ensemble_YYYYMMDD_HHMMSS.csv

# 3. 앙상블 (1분 미만)
python src/ensemble/two_model_ensemble.py
# 출력: submissions/ensemble/ensemble_xgb_deepfm_65_35_MMDD_HHMM.csv
```

**검증**: 원본 제출 파일과 비교
```bash
python compare_predictions.py
```

---

## 🏋️ 전체 재현 (학습부터)

처음부터 모델 학습 후 추론:

### Step 1: 환경 및 데이터 (위와 동일)

### Step 2: 모델 학습

#### XGBoost v6 학습 (2-3시간)

**Step 2-1: 데이터 전처리 (필수, 10-15분)**
```bash
python -m src.xgboost.xgboost_prepare_data
```
이 단계에서 raw parquet 파일을 XGBoost용 numpy 배열로 변환합니다.

**Step 2-2: 학습 실행 (2-3시간)**
```bash
python train/xgboost_train.py
```
**출력**:
- `checkpoints/xgboost/xgboost_kfold_YYYYMMDD_HHMMSS/xgboost_fold_{1-5}.json`
- OOF Score: 0.353-0.354 (예상)

**주요 설정** (src/utils/config.py - XGBOOST_CONFIG_V6):
- n_estimators: 245
- max_depth: 9
- learning_rate: 0.0392
- reg_lambda: 0.32 (Anti-overfitting)
- scale_pos_weight: 51.6
- Random seed: 42

#### DeepFM v2.2 학습 (3-4시간, GPU 필수)
```bash
python train/deepfm_train.py
```
**출력**:
- `checkpoints/deepfm_kfold/deepfm-v2.2-kfold-MMDD_HHMM/fold_{1-5}_model.pth`
- OOF Score: 0.3507 (목표)

**주요 설정** (src/utils/config.py - DEEPFM_CONFIG):
- Embedding: inventory_id 10D, 기타 16D
- Hidden layers: [256, 128]
- Dropout: 0.25
- Learning rate: 0.001 (Cosine Annealing)
- Epochs: 40
- Random seed: 42

### Step 3: 추론 실행 (위 Step 3과 동일)

---

## 📊 재현성 검증 결과

**추론 재현**: ✅ **100% 완벽 재현 확인**
- Max difference: 0.00000000
- 원본과 재생성 파일이 완전히 동일

**검증 항목**:
- ✅ Random seed 42 전역 고정
- ✅ 모든 processor 포함
- ✅ 실제 사용한 추론 스크립트 포함
- ✅ 학습 코드 포함

---

## 📁 주요 파일

### 모델 체크포인트 (50MB)
- `checkpoints/xgboost/xgboost_fold_{1-5}.json` (43MB, 5 모델)
- `checkpoints/deepfm/fold_{1-5}_model.pth` (7MB, 5 모델)

### Processors
- `data/processed/xgboost/processor.pkl` (7.4KB)
- `data/processed/deepfm_new/processor.pkl` (947KB)

### 학습 스크립트
- `train/xgboost_train.py` - XGBoost v6 5-Fold 학습
- `train/deepfm_train.py` - DeepFM v2.2 5-Fold 학습

### 추론 스크립트
- `src/xgboost/xgboost_predict.py` - XGBoost 예측
- `src/deepfm/deepfm_predict_kfold.py` - DeepFM 예측
- `src/ensemble/two_model_ensemble.py` - 2-모델 앙상블

### 문서
- `FINAL_SUBMISSION_README.md` - 상세 가이드
- `report/MODEL_DEVELOPMENT_REPORT.md` - 개발 보고서 (한국어, 50+ 페이지)

---

## ⏱️ 예상 소요 시간

| 작업 | 시간 | 비고 |
|------|------|------|
| 환경 설정 | 10분 | conda env 생성 |
| **추론만** | **20-30분** | **이미 학습된 모델 사용** |
| XGBoost 학습 | 2-3시간 | CPU/GPU |
| DeepFM 학습 | 3-4시간 | GPU 권장 |
| **전체 (학습+추론)** | **6-8시간** | **처음부터 재현** |

---

## 🎯 최종 제출 모델

**모델**: XGBoost v6 (65%) + DeepFM v2.2 (35%)
- **Out-of-Fold Score**: 0.355X (예상)
- **Public LB**: 0.350-0.352 (예상)
- **파일**: `submissions/ensemble_xgb_deepfm_65_35_1012_1728.csv`

---

## 💡 팁

### GPU 사용
DeepFM 학습/추론에 GPU 강력 권장:
- CUDA 11.8+ 필요
- GPU 없으면 CPU로 실행 가능 (10배 느림)

### WandB 로깅
학습 과정 모니터링 (선택사항):
```bash
# WandB 로그인
wandb login

# 학습 실행 시 자동으로 로깅됨
```

### 메모리 부족 시
DeepFM batch size 조정 (src/utils/config.py):
```python
"BATCH_SIZE": 512,  # 1024 → 512
```

---

## ❓ 문의

제출 패키지 관련 질문은 대회 운영진에게 문의 바랍니다.

**작성일**: 2025-10-17
**대회**: 토스 NEXT ML CHALLENGE
