import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import wandb
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.config import (
    XGBOOST_CONFIG,
    XGBOOST_CONFIG_V6,
    XGBOOST_WANDB_CONFIG,
    KFOLD_CONFIG,
)
from src.utils.metrics import calculate_competition_metrics

# 🎯 v6 실험: 균형잡힌 과적합 방지 (Target: Public 0.351-0.352)
USE_V6_CONFIG = True  # True: v6 (균형), False: v4 (현재)
CONFIG = XGBOOST_CONFIG_V6 if USE_V6_CONFIG else XGBOOST_CONFIG


# WandB 콜백 제거 - fold 완료 시점에만 핵심 메트릭 로깅으로 단순화


# Constants
DATA_DIR = "data/processed/xgboost"
REQUIRED_FILES = [
    "X_train.npy",
    "X_val.npy",
    "y_train.npy",
    "y_val.npy",
    "metadata.pkl",
    "processor.pkl",
    "feature_names.pkl",
]
CHECKPOINT_DIR = "checkpoints/xgboost"


def load_prepared_data():
    """Load prepared training and validation data"""
    print("Loading prepared data...")

    # 데이터 파일 존재 확인
    for file in REQUIRED_FILES:
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
            np.load(f"{DATA_DIR}/X_train.npy"),
            columns=feature_names
        )
        val_X = pd.DataFrame(
            np.load(f"{DATA_DIR}/X_val.npy"),
            columns=feature_names
        )
        train_y = pd.Series(np.load(f"{DATA_DIR}/y_train.npy"))
        val_y = pd.Series(np.load(f"{DATA_DIR}/y_val.npy"))

    except Exception as e:
        raise RuntimeError(f"Failed to load prepared data: {e}")

    print(f"Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"Features: {len(feature_names)}")

    return train_X, train_y, val_X, val_y


def train_kfold_xgboost(X, y, val_X, val_y, config, kfold_config):
    """Train XGBoost with K-Fold Cross Validation"""

    # Initialize K-Fold
    kf = StratifiedKFold(
        n_splits=kfold_config["n_splits"],
        shuffle=kfold_config["shuffle"],
        random_state=kfold_config["random_state"],
    )

    models = []
    fold_scores = []
    val_predictions = np.zeros(len(val_y))
    oof_predictions = np.zeros(len(y))

    print(f"Starting {kfold_config['n_splits']}-Fold Cross Validation...")

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold + 1}/{kfold_config['n_splits']}")

        # Split data for this fold
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # Train model for this fold
        print(f"Training fold {fold + 1}...")
        watchlist = [(dtrain, "train"), (dvalid, "valid")]

        # WandB 콜백은 제거하고 fold 완료 시에만 로깅 (더 깔끔한 차트)
        model = xgb.train(
            config,
            dtrain,
            num_boost_round=config["n_estimators"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=30,  # Optuna 최종 학습과 동일
            verbose_eval=50,
        )

        models.append(model)

        # Predict on validation fold (out-of-fold) - 한 번만 예측
        fold_pred = model.predict(dvalid)
        oof_predictions[valid_idx] = fold_pred

        # Predict on test set
        dval = xgb.DMatrix(val_X)
        val_predictions += model.predict(dval) / kfold_config["n_splits"]

        # Calculate fold metrics - 이미 계산된 fold_pred 재사용
        fold_auc = roc_auc_score(y_valid_fold, fold_pred)
        fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
        fold_comp_score = fold_metrics["comp_score"]

        fold_scores.append(
            {"fold": fold + 1, "auc": fold_auc, "comp_score": fold_comp_score}
        )

        print(f"   Fold {fold + 1} AUC: {fold_auc:.6f}")
        print(f"   Fold {fold + 1} Competition Score: {fold_comp_score:.6f}")

        # 각 fold 완료 후 WandB에 로깅 - 통합 메트릭으로
        wandb.log(
            {
                "fold_auc": fold_auc,
                "fold_comp_score": fold_comp_score,
                "best_iteration": model.best_iteration,
                "completed_folds": fold + 1,
                "fold_number": fold + 1,
            }
        )

    return models, oof_predictions, val_predictions, fold_scores


def main():
    start_time = time.time()

    try:
        print("Starting XGBoost K-Fold training...")

        # Load data
        data_start = time.time()
        train_X, train_y, val_X, val_y = load_prepared_data()
        data_time = time.time() - data_start
        print(f"Data loaded in {data_time:.2f} seconds")
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # Initialize WandB
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"xgboost-kfold-{timestamp}"

    wandb.init(
        project=XGBOOST_WANDB_CONFIG["PROJECT"],
        entity=XGBOOST_WANDB_CONFIG["ENTITY"],
        name=run_name,
        tags=XGBOOST_WANDB_CONFIG["TAGS"] + ["kfold"],
        notes="XGBoost with K-Fold Cross Validation",
        config={
            **CONFIG,
            **KFOLD_CONFIG,
            "data_load_time": data_time,
            "num_features": train_X.shape[1],
            "train_samples": len(train_X),
            "val_samples": len(val_X),
            "train_ctr": train_y.mean(),
            "val_ctr": val_y.mean(),
            "xgboost_version": "v6" if USE_V6_CONFIG else "v4",
        },
    )

    # Train K-Fold models
    train_start = time.time()
    models, oof_predictions, val_predictions, fold_scores = train_kfold_xgboost(
        train_X, train_y, val_X, val_y, CONFIG, KFOLD_CONFIG
    )
    train_time = time.time() - train_start

    print("\nEvaluating K-Fold results...")

    # Calculate Out-of-Fold metrics
    oof_comp_metrics = calculate_competition_metrics(train_y, oof_predictions)
    oof_metrics = {
        "oof_auc": oof_comp_metrics["auc"],
        "oof_ap": oof_comp_metrics["ap"],
        "oof_comp_score": oof_comp_metrics["comp_score"],
        "oof_wll": oof_comp_metrics["wll"],
    }

    # Calculate validation metrics
    val_comp_metrics = calculate_competition_metrics(val_y, val_predictions)
    val_metrics = {
        "val_auc": val_comp_metrics["auc"],
        "val_ap": val_comp_metrics["ap"],
        "val_comp_score": val_comp_metrics["comp_score"],
        "val_wll": val_comp_metrics["wll"],
    }

    # Print fold results
    print(f"\nIndividual Fold Results:")
    for fold_result in fold_scores:
        print(
            f"  Fold {fold_result['fold']}: AUC={fold_result['auc']:.6f}, CompScore={fold_result['comp_score']:.6f}"
        )

    # Calculate fold statistics
    fold_aucs = [f["auc"] for f in fold_scores]
    fold_comp_scores = [f["comp_score"] for f in fold_scores]

    print(f"\nCross-Validation Results:")
    print(f"  CV AUC: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
    print(
        f"  CV Competition Score: {np.mean(fold_comp_scores):.6f} ± {np.std(fold_comp_scores):.6f}"
    )

    print(f"\nOut-of-Fold Metrics:")
    print(f"  OOF AUC: {oof_metrics['oof_auc']:.6f}")
    print(f"  OOF Competition Score: {oof_metrics['oof_comp_score']:.6f}")

    print(f"\nValidation Metrics (Ensemble):")
    print(f"  Val AUC: {val_metrics['val_auc']:.6f}")
    print(f"  Val Competition Score: {val_metrics['val_comp_score']:.6f}")

    # Save models
    model_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(model_dir, exist_ok=True)

    for i, model in enumerate(models):
        model.save_model(f"{model_dir}/xgboost_fold_{i+1}.json")

    # Save predictions separately
    pd.DataFrame({"oof_predictions": oof_predictions}).to_parquet(
        f"{model_dir}/oof_predictions.parquet"
    )
    pd.DataFrame({"val_predictions": val_predictions}).to_parquet(
        f"{model_dir}/val_predictions.parquet"
    )

    # 핵심 메트릭만 WandB에 로깅 (차트 깔끔하게)
    wandb.log(
        {
            # 성능 메트릭 (가장 중요)
            "final_val_comp_score": val_metrics["val_comp_score"],
            "final_val_auc": val_metrics["val_auc"],
            "final_cv_comp_score": np.mean(fold_comp_scores),
            "final_cv_auc": np.mean(fold_aucs),
            "cv_std": np.std(fold_comp_scores),

            # 시간 메트릭
            "training_time_minutes": train_time / 60,
            "total_time_minutes": (time.time() - start_time) / 60,

            # 목표 달성 여부
            "target_achieved": val_metrics["val_comp_score"] >= 0.35,
        }
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("XGBoost K-Fold Training Completed!")
    print("=" * 50)
    print(f"Data Load Time: {data_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Models saved: {model_dir}")

    # Target achievement check
    target_score = 0.35
    achieved = val_metrics["val_comp_score"] >= target_score
    print(f"Target: {target_score}+ (current: {val_metrics['val_comp_score']:.6f})")
    if achieved:
        print("TARGET ACHIEVED with K-Fold ensemble!")

    wandb.finish()


if __name__ == "__main__":
    main()
