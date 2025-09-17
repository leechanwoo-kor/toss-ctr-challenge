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
