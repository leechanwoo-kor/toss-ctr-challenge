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

# DeepFM 현업 표준 설정 (Google/Meta/ByteDance 기준)
DEEPFM_CONFIG = {
    "SAMPLE_SIZE": None,  # 전체 데이터 사용 (10.7M)
    "BATCH_SIZE": 1024,  # 현업 표준 (더 큰 배치)
    "EPOCHS": 30,  # 충분한 학습을 위해 증가 (15→30)
    "LEARNING_RATE": 0.001,  # 원래대로 복원 (0.003→0.001)
    "EMBEDDING_DIM": 16,  # 최적화: 가벼운 임베딩 (32→16)
    "HIDDEN_DIMS": [256, 128],  # 최적화: 가벼운 네트워크 ([1024,512,256]→[256,128])
    "DROPOUT": 0.2,  # 현업 표준 (낮은 드롭아웃)
    "WEIGHT_DECAY": 1e-5,  # 현업 표준
    "POS_WEIGHT": 51.6,  # 원래대로 복원 (5.0→51.6)
    "FOCAL_ALPHA": 0.25,  # Focal Loss alpha parameter
    "FOCAL_GAMMA": 2.0,  # Focal Loss gamma parameter
    "SEED": 42,
    "EARLY_STOPPING": 5,
    "PATIENCE": 3,  # Early stopping patience
}

# WandB 설정
WANDB_CONFIG = {
    "PROJECT": "toss-ctr-deepfm",
    "ENTITY": None,
    "TAGS": ["deepfm", "industry-standard", "focal-loss", "full-dataset"],
    "NOTES": "DeepFM with industry standard hyperparams + Focal Loss",
}

# LightGBM 수정된 설정 (불균형 데이터 처리 개선)
LIGHTGBM_CONFIG = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "device": "cpu",                     # CPU로 안정성 확보
    "num_leaves": 127,                   # 안정적인 설정
    "learning_rate": 0.1,                # 0.05 → 0.1 (더 적극적 학습)
    "feature_fraction": 0.8,             # 과적합 방지
    "bagging_fraction": 0.8,             # 과적합 방지
    "bagging_freq": 5,                   # 더 안정적으로
    "min_child_samples": 20,             # 50 → 20 (덜 제한적)
    "min_child_weight": 0.001,           # 불균형 데이터 대응
    "reg_alpha": 0.1,                    # L1 정규화
    "reg_lambda": 1.0,                   # L2 정규화 강화
    "max_depth": 8,                      # XGBoost와 동일
    "num_boost_round": 1000,             # 적절한 수준으로 조정
    "early_stopping_rounds": 100,        # 50 → 100 (더 관대한 조기 종료)
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
    "is_unbalance": True,                # LightGBM 내장 불균형 처리 활성화 (scale_pos_weight 대신)
    "force_col_wise": True,              # 메모리 효율성
    "bin_construct_sample_cnt": 200000,  # 안정적인 이진화
}

# LightGBM WandB 설정
LIGHTGBM_WANDB_CONFIG = {
    "PROJECT": "toss-ctr-lightgbm",
    "ENTITY": None,
    "TAGS": ["lightgbm", "tree-based", "industry-standard", "full-dataset"],
    "NOTES": "LightGBM with industry standard hyperparams for CTR prediction",
}

# XGBoost Optuna 최적화 설정 (0.355654 달성 - 9/24 Trial 75)
XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "booster": "gbtree",
    "device": "cuda:0",                  # GPU 가속
    "tree_method": "hist",               # 히스토그램 방식
    "max_depth": 8,                      # Trial 75 최적값
    "learning_rate": 0.05955422983797188, # Trial 75 최적값 (더 안정적)
    "n_estimators": 325,                 # Trial 75 최적값 (충분한 학습)
    "min_child_weight": 169,             # Trial 75 최적값 (강력한 과적합 방지)
    "reg_alpha": 0.03622340097891742,    # Trial 75 최적값 (L1 정규화)
    "reg_lambda": 6.672525788694994,     # Trial 75 최적값 (강력한 L2 정규화)
    "subsample": 0.9763655879223412,     # Trial 75 최적값 (높은 샘플 활용)
    "colsample_bytree": 0.8873768149476484, # Trial 75 최적값 (특성 활용)
    "colsample_bylevel": 0.8904543884294897, # Trial 75 최적값 (레벨별 샘플링)
    "colsample_bynode": 0.8870357675572318,  # Trial 75 최적값 (노드별 샘플링)
    "gamma": 0.6334784828161101,         # Trial 75 최적값 (분할 제약)
    "max_delta_step": 9,                 # Trial 75 최적값 (안정성)
    "scale_pos_weight": 51.6,            # 클래스 불균형 (98.1/1.9)
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 20,
}

# CatBoost 현업 표준 설정 (CTR 최적화)
CATBOOST_CONFIG = {
    "iterations": 300,                   # 트리 수
    "learning_rate": 0.1,                # 학습률
    "depth": 8,                         # 트리 깊이
    "l2_leaf_reg": 3,                   # L2 정규화
    "border_count": 128,                # 범주 경계
    "thread_count": -1,                 # 멀티쓰레딩
    "random_seed": 42,                  # 재현성
    "loss_function": "Logloss",         # 손실함수
    "eval_metric": "AUC",               # 평가지표
    "early_stopping_rounds": 30,        # 조기종료
    "verbose": 50,                      # 로그 출력
    "use_best_model": True,             # 최적 모델 사용
    "task_type": "GPU",                 # GPU 사용
    "class_weights": [1, 51.6],         # 클래스 불균형 (98.1/1.9)
}

# CatBoost WandB 설정
CATBOOST_WANDB_CONFIG = {
    "PROJECT": "toss-ctr-catboost",
    "ENTITY": None,
    "TAGS": ["catboost", "tree-based", "industry-standard", "full-dataset"],
    "NOTES": "CatBoost with industry standard hyperparams for CTR prediction",
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

# 카테고리 피처 정의 (메인 config에서 import)
# CATEGORICAL_FEATURES는 src.config에서 정의됨 - 중복 제거를 위해 삭제

# 제외할 컬럼
EXCLUDE_COLUMNS = {"clicked", "seq", "ID"}
