# Experimental Scripts

이 디렉토리는 분석 및 실험용 스크립트를 포함합니다.

## Scripts

### `xgboost_optuna.py`
- **목적**: XGBoost 하이퍼파라미터 베이지안 최적화
- **사용법**: `python experiments/xgboost_optuna.py`
- **설명**: Optuna를 사용하여 XGBoost 모델의 최적 하이퍼파라미터를 탐색합니다.

### `analyze_feature_importance.py`
- **목적**: K-fold XGBoost 모델의 feature importance 분석
- **사용법**: `python experiments/analyze_feature_importance.py`
- **설명**: 5-fold 모델들의 피처 중요도를 분석하고 시각화합니다.

## Note

이 스크립트들은 메인 학습/추론 파이프라인과 독립적으로 실행됩니다.
프로덕션 코드가 아닌 분석 및 튜닝 목적으로만 사용하세요.
