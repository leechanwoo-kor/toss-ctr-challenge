"""
DeepFM 설정 관리
"""

# DeepFM 현업 표준 설정 (Google/Meta/ByteDance 기준)
DEEPFM_CONFIG = {
    "SAMPLE_SIZE": None,  # 전체 데이터 사용 (10.7M)
    "BATCH_SIZE": 1024,  # 현업 표준 (더 큰 배치)
    "EPOCHS": 10,
    "LEARNING_RATE": 0.001,  # 현업 표준 (빠른 수렴)
    "EMBEDDING_DIM": 32,  # 현업 표준 (더 큰 임베딩)
    "HIDDEN_UNITS": [1024, 512, 256],  # 현업 표준 (더 큰 용량)
    "DROPOUT": 0.2,  # 현업 표준 (낮은 드롭아웃)
    "WEIGHT_DECAY": 1e-5,  # 현업 표준
    "SEED": 42,
    "EARLY_STOPPING": 5,
}

# WandB 설정
WANDB_CONFIG = {
    "PROJECT": "toss-ctr-deepfm",
    "ENTITY": None,
    "TAGS": ["deepfm", "industry-standard", "focal-loss", "full-dataset"],
    "NOTES": "DeepFM with industry standard hyperparams + Focal Loss",
}

# LightGBM 현업 표준 설정 (CTR 최적화) - TreeBasedFeatureProcessor 적용 후 재조정
LIGHTGBM_CONFIG = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "device": "cpu",                     # GPU 문제 가능성으로 일시적 CPU 사용
    # "gpu_platform_id": 0,
    # "gpu_device_id": 0,
    "num_leaves": 31,                    # 기본값으로 복원 (127→31)
    "learning_rate": 0.1,                # 학습률 증가 (0.05→0.1)
    "feature_fraction": 0.9,             # 피처 활용도 증가 (0.8→0.9)
    "bagging_fraction": 0.9,             # 안정성 증가 (0.8→0.9)
    "bagging_freq": 1,                   # 매 iteration (5→1)
    "min_child_samples": 20,             # 기본값으로 복원 (100→20)
    "min_child_weight": 0.001,           # 불균형 데이터 대응
    "reg_alpha": 0.1,                    # L1 정규화 추가
    "reg_lambda": 0.1,                   # L2 정규화 추가
    "max_depth": -1,                     # 제한 없음 (num_leaves로 제어)
    "num_boost_round": 2000,             # 트리 수 증가 (1000→2000)
    "early_stopping_rounds": 100,        # 조기 종료 완화 (50→100)
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# LightGBM WandB 설정
LIGHTGBM_WANDB_CONFIG = {
    "PROJECT": "toss-ctr-lightgbm",
    "ENTITY": None,
    "TAGS": ["lightgbm", "tree-based", "industry-standard", "full-dataset"],
    "NOTES": "LightGBM with industry standard hyperparams for CTR prediction",
}

# XGBoost 현업 표준 설정 (Netflix/Uber 기준)
XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "device": "cuda:0",                  # GPU 가속 (새로운 방식)
    "tree_method": "hist",               # 히스토그램 방식
    "max_depth": 8,                      # 깊이 제한
    "learning_rate": 0.1,               # 안정적 수렴
    "n_estimators": 200,                # 충분한 트리 수
    "subsample": 0.9,                    # 행 샘플링
    "colsample_bytree": 0.9,             # 열 샘플링
    "colsample_bylevel": 0.9,            # 레벨별 열 샘플링
    "min_child_weight": 100,             # 큰 데이터셋용
    "reg_alpha": 0.1,                    # L1 정규화
    "reg_lambda": 0.1,                   # L2 정규화
    "scale_pos_weight": 51.6,            # 클래스 불균형 (98.1/1.9)
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 20,
}

# K-Fold Cross Validation 설정
KFOLD_CONFIG = {
    "n_splits": 5,                       # 5-fold CV
    "shuffle": True,                     # 데이터 셔플
    "random_state": 42,                  # 재현성
    "stratify": True,                    # 층화 표본추출
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

# 카테고리 피처 정의
CATEGORICAL_FEATURES = ["gender", "age_group", "inventory_id", "day_of_week", "hour"]

# 제외할 컬럼
EXCLUDE_COLUMNS = {"clicked", "seq", "ID"}
