"""
모델별 하이퍼파라미터 설정 - Toss CTR Challenge

역할:
- 모델별 하이퍼파라미터 (DeepFM, XGBoost)
- K-Fold 교차검증 설정
- 제외할 컬럼 정의 (EXCLUDE_COLUMNS)
"""

DEEPFM_CONFIG = {
    "SAMPLE_SIZE": None,
    "BATCH_SIZE": 1024,
    "EPOCHS": 40,
    "LEARNING_RATE": 0.001,
    "EMBEDDING_DIM": 16,
    "HIDDEN_DIMS": [256, 128],
    "DROPOUT": 0.25,
    "WEIGHT_DECAY": 2e-5,
    "POS_WEIGHT": 51.6,
    "SEED": 42,
    "EARLY_STOPPING": 7,
    "SCHEDULER_TYPE": "cosine",
    "COSINE_T_MAX": 40,
    "COSINE_T0": 5,
    "COSINE_ETA_MIN_RATIO": 0.01,
    "WANDB_LOG_INDIVIDUAL_FOLDS": False
}

DEEPFM_DYNAMIC_CONFIG = {
    "SAMPLE_SIZE": None,
    "BATCH_SIZE": 1024,
    "EPOCHS": 40,
    "LEARNING_RATE": 0.001,
    "EMBEDDING_DIM": 16,
    "HIDDEN_DIMS": [256, 128],
    "DROPOUT": 0.25,
    "WEIGHT_DECAY": 2e-5,
    "POS_WEIGHT": 51.6,
    "SEED": 42,
    "EARLY_STOPPING": 7,
    "USE_DYNAMIC_EMBEDDING": True,
    "MIN_EMBED_DIM": 4,
    "MAX_EMBED_DIM": 50,
    "SCHEDULER_TYPE": "cosine",
    "COSINE_T_MAX": 40,
    "COSINE_T0": 5,
    "COSINE_ETA_MIN_RATIO": 0.01,
    "WANDB_LOG_INDIVIDUAL_FOLDS": False,
}

XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "device": "cuda:0",
    "tree_method": "hist",
    "max_depth": 9,
    "learning_rate": 0.03917586621687341,
    "n_estimators": 280,
    "min_child_weight": 248,
    "reg_alpha": 0.27450205788134485,
    "reg_lambda": 0.2844933895906582,
    "subsample": 0.8800090828918387,
    "colsample_bytree": 0.7222136942580359,
    "colsample_bylevel": 0.7408740593918586,
    "colsample_bynode": 0.7522542036638713,
    "gamma": 6.806264992795538,
    "max_delta_step": 3,
    "scale_pos_weight": 51.6,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 20
}

XGBOOST_CONFIG_V6 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "device": "cuda:0",
    "tree_method": "hist",
    "max_depth": 9,
    "learning_rate": 0.03917586621687341,
    "n_estimators": 245,
    "min_child_weight": 265,
    "reg_alpha": 0.27450205788134485,
    "reg_lambda": 0.32,
    "subsample": 0.8800090828918387,
    "colsample_bytree": 0.7222136942580359,
    "colsample_bylevel": 0.7408740593918586,
    "colsample_bynode": 0.7522542036638713,
    "gamma": 7.6,
    "max_delta_step": 3,
    "scale_pos_weight": 51.6,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 18
}

KFOLD_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "stratify": True,
}

EXCLUDE_COLUMNS = {"clicked", "seq", "ID"}
