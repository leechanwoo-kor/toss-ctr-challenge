"""
DeepFM 설정 관리
"""

# DeepFM 최적 설정 (Optuna Trial 2 결과)
DEEPFM_CONFIG = {
    "SAMPLE_SIZE": None,  # 전체 데이터 사용 (10.7M)
    "BATCH_SIZE": 512,  # Optuna 최적값
    "EPOCHS": 10,
    "LEARNING_RATE": 0.000384,  # Optuna 최적값 (3.84e-4)
    "EMBEDDING_DIM": 24,  # Optuna 최적값
    "HIDDEN_UNITS": [512, 256, 128],  # Optuna 최적값
    "DROPOUT": 0.337,  # Optuna 최적값
    "WEIGHT_DECAY": 1.57e-6,  # Optuna 최적값
    "SEED": 42,
    "EARLY_STOPPING": 5,
}

# WandB 설정
WANDB_CONFIG = {
    "PROJECT": "toss-ctr-deepfm",
    "ENTITY": None,
    "TAGS": ["deepfm", "improved-preprocessing", "tuned-hyperparams", "full-dataset"],
    "NOTES": "DeepFM with improved preprocessing + tuned hyperparams",
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
