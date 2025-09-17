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

# LightGBM 현업 표준 설정 (Airbnb/Booking.com 기준)
LIGHTGBM_CONFIG = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,                    # 현업 표준
    "learning_rate": 0.1,                # 현업 표준
    "feature_fraction": 0.8,             # 과적합 방지
    "bagging_fraction": 0.8,             # 안정성
    "bagging_freq": 5,
    "min_child_samples": 20,             # 불균형 데이터 대응
    "num_boost_round": 1000,             # 충분한 트리 수
    "early_stopping_rounds": 50,         # 조기 종료
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,                        # 모든 CPU 사용
    "class_weight": "balanced",          # 불균형 처리
}

# LightGBM WandB 설정
LIGHTGBM_WANDB_CONFIG = {
    "PROJECT": "toss-ctr-lightgbm",
    "ENTITY": None,
    "TAGS": ["lightgbm", "tree-based", "industry-standard", "full-dataset"],
    "NOTES": "LightGBM with industry standard hyperparams for CTR prediction",
}

# Optuna 설정
OPTUNA_CONFIG = {
    "STUDY_NAME": "deepfm-hyperopt",
    "N_TRIALS": 50,
    "EPOCHS_PER_TRIAL": 3,
    "EARLY_STOPPING": 2,
    "SAMPLE_SIZE": 2000000,
    "TIMEOUT": 3600 * 12,
}

# 카테고리 피처 정의
CATEGORICAL_FEATURES = ["gender", "age_group", "inventory_id", "day_of_week", "hour"]

# 제외할 컬럼
EXCLUDE_COLUMNS = {"clicked", "seq", "ID"}
