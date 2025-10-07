import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import wandb

# Optuna XGBoost warning 숨기기
warnings.filterwarnings("ignore", message="The reported value is ignored")

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.metrics import calculate_competition_metrics
from src.utils.config import XGBOOST_CONFIG, XGBOOST_WANDB_CONFIG, KFOLD_CONFIG

# Constants
DATA_DIR = "data/processed/xgboost"
CHECKPOINT_DIR = "checkpoints/xgboost"
OPTUNA_STUDIES_DIR = "optuna_studies"
TARGET_SCORE = 0.350  # v3 목표: Public LB 0.350+ (현재 0.3486)
N_TRIALS = 200  # v3 확장 탐색: 더 많은 trials로 breakthrough 찾기
TIMEOUT_HOURS = 12  # 12시간 집중 탐색 (충분한 탐색 시간)


def load_prepared_data():
    """전처리된 데이터 로드"""
    print("Loading prepared data...")

    # 데이터 파일 존재 확인
    required_files = [
        "X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy",
        "metadata.pkl", "processor.pkl", "feature_names.pkl"
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(DATA_DIR, file)):
            raise FileNotFoundError(
                f"Required file not found: {file}. Run 'python -m src.xgboost.xgboost_prepare_data' first."
            )

    try:
        # 피처명 먼저 로드
        with open(f"{DATA_DIR}/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # 데이터를 바로 DataFrame으로 로드 (메모리 효율적)
        train_X = pd.DataFrame(
            np.load(f"{DATA_DIR}/X_train.npy"), columns=feature_names
        )
        val_X = pd.DataFrame(
            np.load(f"{DATA_DIR}/X_val.npy"), columns=feature_names
        )
        train_y = pd.Series(np.load(f"{DATA_DIR}/y_train.npy"))
        val_y = pd.Series(np.load(f"{DATA_DIR}/y_val.npy"))

        print(f"Train: {train_X.shape}, Val: {val_X.shape}")
        print(f"Features: {len(feature_names)}")

        return train_X, train_y, val_X, val_y, feature_names

    except Exception as e:
        raise RuntimeError(f"Failed to load prepared data: {e}")


def objective(trial):
    """Optuna 목적 함수 - K-fold CV로 하이퍼파라미터 최적화"""

    print(f"\nTrial {trial.number} starting...")

    # 🎯 v4 Anti-Overfitting 전략 (Validation-LB 간극 최소화)
    # Problem: v3는 Val 0.356인데 LB 0.348 (gap -0.008) - 과적합!
    # Strategy: 학습량 대폭 감소 + 정규화 강화 → LB 성능 개선
    # Goal: Val-LB 간극을 -0.002 이내로 줄여서 LB 0.350+ 달성
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "device": "cuda:0",
        "tree_method": "hist",

        # Core hyperparameters - 과적합 방지
        "max_depth": trial.suggest_int("max_depth", 7, 9),  # 11 → 9 (깊은 트리 제한)
        "learning_rate": trial.suggest_float("learning_rate", 0.020, 0.040, log=False),  # 낮은 범위로 이동
        "n_estimators": trial.suggest_int("n_estimators", 150, 280, step=10),  # 550 → 280 (절반 이하!)

        # Regularization - 강화된 탐색 (과적합 방지 핵심)
        "min_child_weight": trial.suggest_int("min_child_weight", 200, 280),  # 더 보수적으로
        "reg_alpha": trial.suggest_float("reg_alpha", 0.25, 0.55),  # L1 강화 (0.35 → 0.55)
        "reg_lambda": trial.suggest_float("reg_lambda", 0.18, 0.40),  # L2 강화 (0.25 → 0.40)

        # Sampling parameters - 다양성 증가 (과적합 방지)
        "subsample": trial.suggest_float("subsample", 0.88, 0.96),  # 약간 낮춤 (더 강한 배깅)
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.80),  # 낮춤 (피처 다양성)
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.65, 0.80),  # 낮춤
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.70, 0.85),  # 낮춤

        # Advanced parameters - 보수적 탐색
        "gamma": trial.suggest_float("gamma", 5.0, 8.0),  # 강화 (6.5 → 8.0)
        "max_delta_step": trial.suggest_int("max_delta_step", 3, 5),  # 안정적 범위

        # Fixed parameters
        "scale_pos_weight": 51.6,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # K-Fold 설정
    kf = StratifiedKFold(
        n_splits=KFOLD_CONFIG["n_splits"],
        shuffle=KFOLD_CONFIG["shuffle"],
        random_state=KFOLD_CONFIG["random_state"],
    )

    cv_scores = []

    # K-fold CV 실행
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"    Training fold {fold + 1}/{KFOLD_CONFIG['n_splits']}...")

        # Split data for this fold
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_valid_fold = X_train.iloc[valid_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # More aggressive pruning for efficiency
        pruning_callback = XGBoostPruningCallback(trial, "valid-logloss")

        # Train model with early stopping
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params["n_estimators"],
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=20,  # v4: 엄격한 early stopping (50 → 20)
                verbose_eval=False,
                callbacks=[pruning_callback],
            )

            # Predict and evaluate (재사용된 dvalid 사용)
            fold_pred = model.predict(dvalid)

            # Competition score 계산
            fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
            cv_scores.append(fold_metrics["comp_score"])

        except optuna.TrialPruned:
            # Pruned trial - 조기 종료
            raise optuna.TrialPruned()
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            # 실패한 fold는 평균값으로 대체
            cv_scores.append(0.32)  # 보수적 점수

    # K-fold CV 평균 점수
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # 🎯 개선된 목적 함수 - 성능과 안정성의 균형
    # 표준편차 페널티를 더 합리적으로 조정
    stability_penalty = cv_std * 0.5  # 안정성 페널티 감소

    objective_score = cv_mean - stability_penalty

    # 깔끔한 WandB 로깅 - 핵심 메트릭만
    wandb.log(
        {
            "trial_cv_score": cv_mean,
            "trial_cv_std": cv_std,
            "trial_objective": objective_score,
            "trial_penalty": stability_penalty,
            "trial_number": trial.number,
            # 핵심 하이퍼파라미터만 로깅
            "max_depth": trial.params["max_depth"],
            "learning_rate": trial.params["learning_rate"],
            "n_estimators": trial.params["n_estimators"],
        }
    )

    print(
        f"Trial {trial.number}: CV={cv_mean:.6f}±{cv_std:.6f}, Obj={objective_score:.6f}"
    )

    return objective_score


def main():
    """메인 함수"""
    global X_train, y_train, X_val, y_val

    print("Starting XGBoost K-Fold Optuna Optimization...")

    try:
        # 데이터 로드
        start_time = time.time()
        X_train, y_train, X_val, y_val, feature_names = load_prepared_data()
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # WandB 초기화
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"xgboost-optuna-kfold-{timestamp}"

    wandb.init(
        project=f"{XGBOOST_WANDB_CONFIG['PROJECT']}-optuna",
        entity=XGBOOST_WANDB_CONFIG["ENTITY"],
        name=run_name,
        tags=XGBOOST_WANDB_CONFIG["TAGS"] + ["optuna", "kfold", "hyperopt"],
        notes="XGBoost K-Fold hyperparameter optimization with Optuna",
        config={
            "num_features": len(feature_names),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_ctr": y_train.mean(),
            "val_ctr": y_val.mean(),
            "kfold_splits": KFOLD_CONFIG["n_splits"],
        },
    )

    # Optuna study 생성 - 개선된 설정
    study_name = f"xgboost-kfold-{timestamp}"
    storage = f"sqlite:///{OPTUNA_STUDIES_DIR}/{study_name}.db"
    os.makedirs(OPTUNA_STUDIES_DIR, exist_ok=True)

    # Fine-tuning에 최적화된 pruning 설정
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Competition score 최대화
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15,    # v2 근처 탐색이므로 더 많은 startup (10→15)
            n_warmup_steps=30,      # Fine-tuning이므로 더 빠른 수렴 가능
            interval_steps=15       # 적당한 pruning 빈도
        ),
        load_if_exists=True,
    )

    print(f"Optuna Study: {study_name}")
    print(f"Target: {TARGET_SCORE}+ Competition Score")
    print(f"K-Fold: {KFOLD_CONFIG['n_splits']}-fold Cross Validation")

    # 최적화 실행
    print(f"Starting v2 fine-tuning optimization with {N_TRIALS} trials...")
    print(f"Strategy: ±10~20% exploration around v2 (Public LB 0.3486)")
    print(f"Key changes: reg_alpha 0.15~0.28, reg_lambda 0.12~0.20, subsample 0.94~0.99")
    print(f"Timeout: {TIMEOUT_HOURS} hours")

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=3600 * TIMEOUT_HOURS,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    # 최적 결과 검증
    if len(study.trials) == 0:
        print("No trials completed successfully")
        wandb.finish()
        return

    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best CV Score: {best_trial.value:.6f}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # 최적 파라미터로 최종 학습 및 평가
    print(f"\nTraining final model with best parameters...")

    best_params = best_trial.params.copy()
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "device": "cuda:0",
            "tree_method": "hist",
            "scale_pos_weight": 51.6,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
    )

    # 최종 K-fold 학습
    kf = StratifiedKFold(
        n_splits=KFOLD_CONFIG["n_splits"],
        shuffle=KFOLD_CONFIG["shuffle"],
        random_state=KFOLD_CONFIG["random_state"],
    )

    final_models = []
    final_cv_scores = []
    val_predictions = np.zeros(len(y_val))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Final training fold {fold + 1}/{KFOLD_CONFIG['n_splits']}...")

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_valid_fold = X_train.iloc[valid_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_params["n_estimators"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=30,  # 일관된 early stopping
            verbose_eval=False,
        )

        final_models.append(model)

        # Validation set 예측
        dval = xgb.DMatrix(X_val)
        val_predictions += model.predict(dval) / KFOLD_CONFIG["n_splits"]

        # Fold 성능
        fold_pred = model.predict(dvalid)
        fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
        final_cv_scores.append(fold_metrics["comp_score"])

    # 최종 평가
    final_metrics = calculate_competition_metrics(y_val, val_predictions)

    print(f"\nFinal Results:")
    print(f"CV Score: {np.mean(final_cv_scores):.6f} ± {np.std(final_cv_scores):.6f}")
    print(f"Validation Score: {final_metrics['comp_score']:.6f}")
    print(f"Validation AUC: {final_metrics['auc']:.6f}")

    # 결과 저장
    result_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(result_dir, exist_ok=True)

    for i, model in enumerate(final_models):
        model.save_model(f"{result_dir}/xgboost_fold_{i+1}.json")

    # 최적 파라미터 저장
    with open(f"{result_dir}/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

    # Study 정보 저장
    study_df = study.trials_dataframe()
    study_df.to_csv(f"{result_dir}/optuna_trials.csv", index=False)

    # WandB 최종 로깅
    wandb.log(
        {
            "best_cv_score": best_trial.value,
            "final_cv_mean": np.mean(final_cv_scores),
            "final_cv_std": np.std(final_cv_scores),
            "final_val_score": final_metrics["comp_score"],
            "final_val_auc": final_metrics["auc"],
            "n_trials_completed": len(study.trials),
            **{f"best_{key}": value for key, value in best_trial.params.items()},
        }
    )

    wandb.finish()

    target_achieved = final_metrics["comp_score"] >= TARGET_SCORE
    if target_achieved:
        print(f"TARGET ACHIEVED: {final_metrics['comp_score']:.6f} >= {TARGET_SCORE}")
    else:
        print(f"Target: {TARGET_SCORE}+ (current: {final_metrics['comp_score']:.6f})")

    print(f"\nResults saved: {result_dir}")


if __name__ == "__main__":
    main()
