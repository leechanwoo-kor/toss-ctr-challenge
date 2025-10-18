# 🎯 본선 제출 가이드

**대회**: 토스 NEXT ML CHALLENGE - 광고 클릭 예측(CTR) 모델 개발
**제출 마감**: 2025년 10월 17일(금) 10:00
**제출 방법**: dacon@dacon.io 이메일 제출

---

## 📦 제출 패키지 구성

### 1. 코드 (학습 + 추론)

#### ⭐ 추론 코드 (최우선)
- **`inference.py`** - Private Score 재현용 추론 스크립트
  - XGBoost v6 (65%) + DeepFM v2.2 (35%) 앙상블
  - Random Seed 42 고정
  - 실행 시간: 약 10-15분 (GPU)

#### 학습 코드 (재현용)
- **`train/xgboost_train.py`** - XGBoost 학습
- **`train/deepfm_train.py`** - DeepFM 학습
- **`train/deepfm_prepare_data.py`** - DeepFM 데이터 전처리

#### 공통 모듈
- **`src/models/`** - 모델 정의
- **`src/utils/`** - 유틸리티 함수
- **`src/deepfm/`** - DeepFM 전처리

### 2. 모델 가중치

- **`checkpoints/xgboost/`** - XGBoost 5-Fold 모델 (JSON 형식)
  - `xgboost_fold_1~5.json` (각 8.5MB, 총 43MB)

- **`checkpoints/deepfm/`** - DeepFM 5-Fold 모델 (PyTorch)
  - `fold_1~5_model.pth` (각 1.4MB, 총 7MB)

- **`data/processed/deepfm_new/processor.pkl`** - DeepFM 전처리 프로세서 (947KB)

**총 용량**: 51MB

### 3. 환경 설정

- **`environment.yml`** - Anaconda 환경 (전체)
- **`requirements.txt`** - Pip 패키지 목록

### 4. 문서

- **`README.md`** - 실행 가이드
- **`report/model_development_report.pdf`** - 모델 개발 보고서 ⭐

---

## 🚀 빠른 시작 (Private Score 재현)

### 1. 환경 설정

```bash
# Conda 환경 생성
conda env create -f environment.yml
conda activate ctr

# 또는 Pip 사용
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# test.csv를 data/ 디렉토리에 배치
# (대회 페이지에서 다운로드)
cp /path/to/downloaded/test.csv data/test.csv
```

### 3. 추론 실행

```bash
# Private Score 재현
python inference.py
```

**출력**: `submissions/final/ensemble_xgb_deepfm_65_35_final.csv`

---

## 📋 재현성 체크리스트

### ✅ Random Seed 고정
- [x] Python random: 42
- [x] NumPy: 42
- [x] PyTorch: 42
- [x] XGBoost: 내장 (모델 저장 시 고정)

### ✅ 하이퍼파라미터 명시
모든 하이퍼파라미터는 코드에 명시되어 있습니다:
- `src/utils/config.py` - 전역 설정
- `inference.py` - 앙상블 가중치 (65:35)
- 학습 코드 내 주석

### ✅ 모델 가중치
- [x] XGBoost: 5-Fold 모델 (JSON)
- [x] DeepFM: 5-Fold 모델 (PyTorch)
- [x] DeepFM Processor: 전처리 설정

### ✅ 코드 실행 가능
- [x] UTF-8 인코딩
- [x] 상대 경로 사용
- [x] 오류 없이 실행 가능

---

## 🔍 코드 검증

### Data Leakage 체크
✅ **없음**
- 학습 데이터와 테스트 데이터 완전 분리
- K-Fold 검증으로 OOF 예측 사용
- 테스트 데이터는 추론 시에만 사용

### 치팅 요소 체크
✅ **없음**
- 외부 데이터 미사용
- Label 정보 미사용
- 정상적인 피처 엔지니어링 및 모델링만 사용

---

## 📊 성능 요약

### Private 리더보드
- **Private Score**: [Your Private Score]
- **사용 모델**: XGBoost v6 (65%) + DeepFM v2.2 (35%)

### 교차 검증
- **XGBoost OOF**: [Score]
- **DeepFM OOF**: 0.3507
- **앙상블 예상**: [Score]

---

## 📧 제출 방법

### 이메일 제출
- **수신**: dacon@dacon.io
- **제목**: [토스 CTR Challenge] 본선 제출 - [팀명]
- **첨부**:
  1. 전체 코드 (.zip)
  2. 모델 가중치 (.zip 또는 클라우드 링크)
  3. 모델 개발 보고서 (.pdf)

### 압축 파일 생성

```bash
# 제출 패키지 압축
cd ..
zip -r toss_ctr_submission.zip toss_ctr_submission/

# 또는 7z 사용 (더 작은 용량)
7z a -t7z toss_ctr_submission.7z toss_ctr_submission/
```

---

## ❓ FAQ

### Q1. inference.py 실행 시 오류 발생
**A**: 다음을 확인하세요:
1. `data/test.csv` 파일이 있는지 확인
2. Conda 환경이 활성화되어 있는지 확인
3. GPU 메모리 부족 시 배치 사이즈 축소

### Q2. 재현 결과가 원본과 다름
**A**: 부동소수점 오차로 인한 미세한 차이는 정상입니다 (< 1e-6)

### Q3. 학습 코드 실행 방법
**A**: `train/` 디렉토리의 README_TRAIN.md 참조

---

## 📞 문의

- **이메일**: [your.email@example.com]
- **GitHub**: [your-github]

---

**제출일**: 2025년 10월 17일
**팀명**: [Your Team Name]
**참가자**: [Your Name]
