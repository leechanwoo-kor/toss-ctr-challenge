"""
모델별 하이퍼파라미터 설정 - Toss CTR Challenge
각 모델(DeepFM, XGBoost, LightGBM, CatBoost)의 하이퍼파라미터 및 실험 설정 관리

역할:
- 모델별 하이퍼파라미터 (DeepFM, XGBoost, LightGBM, CatBoost)
- WandB 설정
- Optuna 최적화 설정
- K-Fold 교차검증 설정
- 제외할 컬럼 정의 (EXCLUDE_COLUMNS)

기본 프로젝트 설정은 src/config.py 참조
"""

# DeepFM v2.2 검증된 최적 설정 (OOF: 0.3507, LB: 0.3466)
DEEPFM_CONFIG = {
    "SAMPLE_SIZE": None,  # 전체 데이터 사용 (10.7M)
    "BATCH_SIZE": 1024,  # 현업 표준 (더 큰 배치)
    "EPOCHS": 40,  # 확장 학습 (30→40) - 더 깊은 수렴
    "LEARNING_RATE": 0.001,  # 검증된 설정 - 0.351135 달성
    "EMBEDDING_DIM": 16,  # 검증된 설정 (16이 최적)
    "HIDDEN_DIMS": [256, 128],  # 검증된 설정
    "DROPOUT": 0.25,  # 정규화 강화 - 과적합 방지
    "WEIGHT_DECAY": 2e-5,  # L2 패널티
    "POS_WEIGHT": 51.6,  # 클래스 불균형 처리
    "FOCAL_ALPHA": 0.25,  # Focal Loss alpha parameter
    "FOCAL_GAMMA": 2.0,  # Focal Loss gamma parameter
    "SEED": 42,
    "EARLY_STOPPING": 7,  # 더 관대한 조기 종료 (5→7) - 후반부 미세 개선 허용
    # Learning Rate Scheduler 설정
    "SCHEDULER_TYPE": "cosine",  # cosine (안정적), cosine_restart (주기적 리셋), plateau
    "COSINE_T_MAX": 40,  # CosineAnnealingLR T_max (전체 에포크 수와 동일)
    "COSINE_T0": 5,  # CosineAnnealingWarmRestarts T_0 (restart 주기, cosine_restart 사용시)
    "COSINE_ETA_MIN_RATIO": 0.01,  # 최소 LR = LR * ratio
    # WandB 로깅 설정
    "WANDB_LOG_INDIVIDUAL_FOLDS": False,  # False = 깔끔한 차트
}

# DeepFM v2.2 Dynamic Embedding (Target: OOF 0.353+, LB 0.349+)
# v2.2 기반 + 카디널리티 기반 동적 embedding (6 × cardinality^0.25)
# 개선점: 파라미터 효율성, 과적합 방지, 중요 피처에 더 많은 표현력
# 예상 효과: OOF +0.002~0.003, LB +0.002~0.003 (v2.2 대비)
DEEPFM_DYNAMIC_CONFIG = {
    "SAMPLE_SIZE": None,  # 전체 데이터 사용
    "BATCH_SIZE": 1024,  # 현업 표준
    "EPOCHS": 40,  # v2.2와 동일
    "LEARNING_RATE": 0.001,  # v2.2 검증된 설정
    "EMBEDDING_DIM": 16,  # 기본값 (동적 embedding 사용 시 무시됨)
    "HIDDEN_DIMS": [256, 128],  # v2.2 검증된 설정
    "DROPOUT": 0.25,  # v2.2와 동일
    "WEIGHT_DECAY": 2e-5,  # v2.2와 동일
    "POS_WEIGHT": 51.6,  # 클래스 불균형 처리
    "FOCAL_ALPHA": 0.25,  # Focal Loss alpha
    "FOCAL_GAMMA": 2.0,  # Focal Loss gamma
    "SEED": 42,
    "EARLY_STOPPING": 7,  # v2.2와 동일
    # 🎯 Dynamic Embedding 설정 (핵심!)
    "USE_DYNAMIC_EMBEDDING": True,  # 동적 embedding 활성화
    "MIN_EMBED_DIM": 4,  # 최소 embedding 차원
    "MAX_EMBED_DIM": 50,  # 최대 embedding 차원
    # Learning Rate Scheduler 설정
    "SCHEDULER_TYPE": "cosine",
    "COSINE_T_MAX": 40,
    "COSINE_T0": 5,
    "COSINE_ETA_MIN_RATIO": 0.01,
    # WandB 로깅 설정
    "WANDB_LOG_INDIVIDUAL_FOLDS": False,
}

# DeepFM Dynamic Embedding v2 - 학습 강화 (Target: OOF 0.352+, LB 0.348+)
# v1 (OOF 0.35123) 기반 + 학습 과정 최적화
# 개선점: 더 공격적인 LR, 더 긴 학습, 더 관대한 조기 종료
# 예상 효과: OOF +0.001~0.002, LB +0.001~0.002 (v1 대비)
DEEPFM_DYNAMIC_ENHANCED_CONFIG = {
    "SAMPLE_SIZE": None,  # 전체 데이터 사용
    "BATCH_SIZE": 1024,  # v1과 동일
    "EPOCHS": 50,  # 40 → 50 (+25% 더 긴 학습)
    "LEARNING_RATE": 0.0015,  # 0.001 → 0.0015 (+50% 더 공격적)
    "EMBEDDING_DIM": 16,  # 기본값 (동적 embedding 사용 시 무시됨)
    "HIDDEN_DIMS": [256, 128],  # v1과 동일 (검증된 구조)
    "DROPOUT": 0.25,  # v1과 동일
    "WEIGHT_DECAY": 2e-5,  # v1과 동일
    "POS_WEIGHT": 51.6,  # 클래스 불균형 처리
    "FOCAL_ALPHA": 0.25,  # Focal Loss alpha
    "FOCAL_GAMMA": 2.0,  # Focal Loss gamma
    "SEED": 42,
    "EARLY_STOPPING": 7,
    # 🎯 Dynamic Embedding 설정 (v1과 동일)
    "USE_DYNAMIC_EMBEDDING": True,
    "MIN_EMBED_DIM": 4,
    "MAX_EMBED_DIM": 50,
    # Learning Rate Scheduler 설정
    "SCHEDULER_TYPE": "cosine",
    "COSINE_T_MAX": 50,  # EPOCHS와 동일
    "COSINE_T0": 5,
    "COSINE_ETA_MIN_RATIO": 0.01,
    # WandB 로깅 설정
    "WANDB_LOG_INDIVIDUAL_FOLDS": False,
}

# WandB 설정
WANDB_CONFIG = {
    "PROJECT": "toss-ctr-deepfm",
    "ENTITY": None,
    "TAGS": ["deepfm", "industry-standard", "focal-loss", "full-dataset"],
    "NOTES": "DeepFM with industry standard hyperparams + Focal Loss",
}

# XGBoost v4 최적 설정 (Optuna Anti-Overfitting - 20251007)
# CV: 0.355010, Validation: 0.355523, Public LB: 0.348995 ✅ (BEST!)
# Strategy: 학습량 감소 + 정규화 강화로 Validation-LB 간극 최소화 (-0.0065)
XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "device": "cuda:0",
    "tree_method": "hist",
    "max_depth": 9,
    "learning_rate": 0.03917586621687341,
    "n_estimators": 280,  # 400 → 280 (30% 감소, 과적합 방지)
    "min_child_weight": 248,  # 195 → 248 (더 보수적)
    "reg_alpha": 0.27450205788134485,  # L1 정규화 강화
    "reg_lambda": 0.2844933895906582,  # L2 정규화 대폭 강화 (+89%)
    "subsample": 0.8800090828918387,  # 0.976 → 0.880 (더 강한 배깅)
    "colsample_bytree": 0.7222136942580359,  # 피처 샘플링
    "colsample_bylevel": 0.7408740593918586,
    "colsample_bynode": 0.7522542036638713,
    "gamma": 6.806264992795538,  # 4.66 → 6.81 (강화)
    "max_delta_step": 3,  # 5 → 3 (안정성)
    "scale_pos_weight": 51.6,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 20,  # 30 → 20 (더 엄격)
}

# XGBoost v6 균형잡힌 과적합 방지 (20251011)
# v4와 v5의 최적 중간점으로 일반화 극대화
# Strategy: v4(gap -0.006)와 v5(극단적)의 균형점 탐색
# Target: OOF 0.353-0.354, Public 0.351-0.352, Gap -0.002~-0.003
XGBOOST_CONFIG_V6 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "device": "cuda:0",
    "tree_method": "hist",
    # 트리 구조 (v4 검증된 값 유지)
    "max_depth": 9,
    "learning_rate": 0.03917586621687341,
    # 🎯 학습량: v4(280)와 v5(210)의 중간
    "n_estimators": 245,  # 균형점: 충분한 학습 + 과적합 방지
    # 🎯 리프 생성 조건: v4(248)와 v5(285)의 중간
    "min_child_weight": 265,  # 작은 리프 억제, 노이즈 학습 방지
    # 🎯 정규화: v4(0.284)보다 강하게
    "reg_alpha": 0.27450205788134485,  # L1 유지
    "reg_lambda": 0.32,  # L2 강화: v4와 v5의 중간값
    # 샘플링 (v4 검증된 값 유지)
    "subsample": 0.8800090828918387,
    "colsample_bytree": 0.7222136942580359,
    "colsample_bylevel": 0.7408740593918586,
    "colsample_bynode": 0.7522542036638713,
    # 🎯 트리 분할 제약: v4(6.81)와 v5(8.5)의 중간
    "gamma": 7.6,  # 의미있는 분할만 허용
    "max_delta_step": 3,
    "scale_pos_weight": 51.6,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    # 🎯 조기 종료: v4(20)와 v5(15)의 중간
    "early_stopping_rounds": 18,  # 적절한 엄격함
}

# K-Fold Cross Validation 설정
KFOLD_CONFIG = {
    "n_splits": 5,  # 5-fold CV
    "shuffle": True,  # 데이터 셔플
    "random_state": 42,  # 재현성
    "stratify": True,  # 층화 표본추출
}

# XGBoost WandB 설정
XGBOOST_WANDB_CONFIG = {
    "PROJECT": "toss-ctr-xgboost",
    "ENTITY": None,
    "TAGS": ["xgboost", "tree-based", "industry-standard", "full-dataset"],
    "NOTES": "XGBoost with industry standard hyperparams for CTR prediction",
}

# Optuna 설정 (DeepFM용)
OPTUNA_CONFIG = {
    "STUDY_NAME": "deepfm-hyperopt",
    "N_TRIALS": 50,
    "EPOCHS_PER_TRIAL": 3,
    "EARLY_STOPPING": 2,
    "SAMPLE_SIZE": 2000000,
    "TIMEOUT": 3600 * 12,
}

# XGBoost Optuna 설정
XGBOOST_OPTUNA_CONFIG = {
    "STUDY_NAME": "xgboost-hyperopt",
    "N_TRIALS": 50,
    "TIMEOUT": 3600 * 4,  # 4시간
    "PRUNER": "MedianPruner",
    "N_STARTUP_TRIALS": 5,
    "N_WARMUP_STEPS": 100,
}

# 카테고리 피처 정의 (메인 config에서 import)
# CATEGORICAL_FEATURES는 src.config에서 정의됨 - 중복 제거를 위해 삭제

# 제외할 컬럼
EXCLUDE_COLUMNS = {"clicked", "seq", "ID"}
