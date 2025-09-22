# 🎯 Ensemble Master Guide

## 📂 파일 정리 완료

### ✅ 현재 활성 파일
```
src/ensemble/
├── __init__.py
└── ensemble_master.py      # 🌟 All-in-One 앙상블 스크립트
```

### 📦 보관된 파일
```
archive/ensemble_old/       # 구버전 파일들
├── run_3model_ensemble.py
├── stacking_ensemble.py
└── final_ensemble_summary.py
```

## 🚀 사용법

### 기본 사용 (모든 앙상블)
```bash
cd src/ensemble
python ensemble_master.py
```

### 모드별 실행
```bash
cd src/ensemble

# 가중평균만
python ensemble_master.py --mode weighted

# 스태킹만
python ensemble_master.py --mode stacking

# 결과 분석만
python ensemble_master.py --mode analyze

# 출력 디렉토리 지정
python ensemble_master.py --output ../../submissions/my_ensemble
```

## 📊 포함된 앙상블 방법

### 1. 가중평균 앙상블 (6가지)
- `xgb_dominant`: XGB(55%) + CB(30%) + DeepFM(15%) ⭐
- `performance_focused`: XGB(50%) + CB(35%) + DeepFM(15%)
- `balanced`: XGB(45%) + CB(35%) + DeepFM(20%)
- `conservative`: XGB(40%) + CB(40%) + DeepFM(20%)
- `equal_importance`: XGB(35%) + CB(35%) + DeepFM(30%)
- `geometric`: 기하평균 (33% 동일)

### 2. 스태킹 앙상블 (2가지)
- `stacking_lightgbm`: LightGBM 메타러너
- `stacking_logistic`: Logistic Regression 메타러너

## 🎯 현재 최고 성능

```
최고 예상: xgb_dominant (0.3523)
실제 LB: 0.3415 (검증됨)
추천: ensemble_weighted_xgb_dominant_[timestamp].csv
```

## 🔧 모델 파일 경로

스크립트가 자동으로 찾는 기본 경로:
```
XGBoost:  submissions/xgboost/xgboost_kfold_0922_1500.csv
CatBoost: submissions/catboost/submission_catboost_catboost-kfold-0920_0228_20250920_024927.csv
DeepFM:   submissions/deepfm_enhanced_0922_1555_unknown.csv
```

## 📈 다음 단계

1. **DeepFM 개선** 후 새 파일로 업데이트
2. **ensemble_master.py** 실행
3. **가장 높은 예상 점수** 파일 제출

---
*모든 앙상블 기능이 하나로 통합되었습니다! 🎉*