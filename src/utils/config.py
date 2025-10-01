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
    "LEARNING_RATE": 0.001,  # 검증된 설정으로 복원 (0.002→0.001) - 0.351135 달성 설정
    "EMBEDDING_DIM": 16,  # 검증된 설정으로 복원 (20→16)
    "HIDDEN_DIMS": [256, 128],  # 검증된 설정으로 복원 ([512,256,128]→[256,128])
    "DROPOUT": 0.25,  # Phase 2: 정규화 강화 (0.2→0.25) - 과적합 방지
    "WEIGHT_DECAY": 2e-5,  # Phase 2: 정규화 강화 (1e-5→2e-5) - L2 패널티 2배
    "POS_WEIGHT": 51.6,  # 원래대로 복원 (5.0→51.6)
    "FOCAL_ALPHA": 0.25,  # Focal Loss alpha parameter
    "FOCAL_GAMMA": 2.0,  # Focal Loss gamma parameter
    "SEED": 42,
    "EARLY_STOPPING": 5,
    # Learning Rate Scheduler 설정
    "SCHEDULER_TYPE": "cosine",  # cosine (안정적), cosine_restart (주기적 리셋), plateau
    "COSINE_T_MAX": 30,  # CosineAnnealingLR T_max (전체 에포크 수)
    "COSINE_T0": 5,  # CosineAnnealingWarmRestarts T_0 (restart 주기, cosine_restart 사용시)
    "COSINE_ETA_MIN_RATIO": 0.01,  # 최소 LR = LR * ratio
    # WandB 로깅 설정
    "WANDB_LOG_INDIVIDUAL_FOLDS": False,  # False = 깔끔한 차트
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

# XGBoost 검증된 최적 설정 (from xgboost_best_params_20250929.json)
# CV: 0.354923, Validation: 0.355868, Public LB: 0.348568
XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "device": "cuda:0",
    "tree_method": "hist",
    "max_depth": 9,
    "learning_rate": 0.04175612007277342,
    "n_estimators": 400,
    "min_child_weight": 195,
    "reg_alpha": 0.2002740001146326,
    "reg_lambda": 0.15010513818084412,
    "subsample": 0.9762245610286578,
    "colsample_bytree": 0.8176096363827465,
    "colsample_bylevel": 0.77295375493092,
    "colsample_bynode": 0.8455520173555662,
    "gamma": 4.65560411412394,
    "max_delta_step": 5,
    "scale_pos_weight": 51.6,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 30,
}

# CatBoost 과적합 방지 설정 (2025/09/25 v2 - Anti-Overfitting)
CATBOOST_CONFIG = {
    "iterations": 700,                  # 반복 줄이기 (1000 → 700, 과적합 방지)
    "learning_rate": 0.03,              # 학습률 감소 (0.05 → 0.03, 더 천천히 학습)
    "depth": 6,                         # 트리 깊이 감소 (8 → 6, 단순화)
    "l2_leaf_reg": 25,                  # L2 정규화 강화 (10 → 25)
    "border_count": 128,                # 분할점 줄이기 (254 → 128, 덜 정교한 분할)
    "random_strength": 2,               # 랜덤성 증가 (1 → 2, 더 많은 노이즈)
    "thread_count": -1,                 # 멀티쓰레딩
    "random_seed": 42,                  # 재현성
    "loss_function": "Logloss",         # 손실함수
    "eval_metric": "AUC",               # 평가지표
    "early_stopping_rounds": 50,        # 조기종료 엄격화 (100 → 50)
    "verbose": 50,                      # 로그 출력
    "use_best_model": True,             # 최적 모델 사용
    "task_type": "GPU",                 # GPU 사용
    "auto_class_weights": "Balanced",   # 클래스 불균형 자동 처리
    "bootstrap_type": "Bernoulli",      # 부트스트랩 타입
    "subsample": 0.7,                   # 샘플링 비율 강화 (0.8 → 0.7, 더 강한 배깅)
    # "colsample_bylevel": 0.8,         # GPU 모드에서 미지원 - 제거
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
