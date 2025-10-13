"""
XGBoost Training with Feature Selection
Feature importance 기반으로 상위 k% 피처만 선택하여 재훈련
"""

import os
import sys
import time
import argparse
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

from src.utils.config import XGBOOST_CONFIG, XGBOOST_WANDB_CONFIG, KFOLD_CONFIG
from src.utils.metrics import calculate_competition_metrics


# Constants
DATA_DIR = "data/processed/xgboost"
CHECKPOINT_DIR = "checkpoints/xgboost"


def load_feature_importance(importance_csv_path):
    """Feature importance CSV 로드"""
    print(f"📊 Loading feature importance from {importance_csv_path}")
    importance_df = pd.read_csv(importance_csv_path)
    print(f"   Total features: {len(importance_df)}")
    return importance_df


def select_top_features(importance_df, n_features=None, keep_ratio=None):
    """
    상위 N개 또는 상위 k% 피처 선택

    Args:
        importance_df: feature importance DataFrame
        n_features: 선택할 피처 개수 (None이면 keep_ratio 사용)
        keep_ratio: 유지할 피처 비율 (0.0-1.0)
    """
    if n_features is None and keep_ratio is None:
        raise ValueError("Either n_features or keep_ratio must be specified")

    if n_features is None:
        n_features = int(len(importance_df) * keep_ratio)

    selected_features = importance_df.head(n_features)['feature'].tolist()

    total_importance = importance_df['importance'].sum()
    selected_importance = importance_df.head(n_features)['importance'].sum()
    importance_retained = (selected_importance / total_importance) * 100

    print(f"\n✅ Feature Selection Complete:")
    print(f"   Selected: {n_features} features ({len(selected_features)/len(importance_df)*100:.1f}% of total)")
    print(f"   Importance retained: {importance_retained:.2f}%")
    print(f"   Removed: {len(importance_df) - n_features} features")

    return selected_features, importance_retained


def load_prepared_data(selected_features=None):
    """Load prepared training and validation data with feature selection"""
    print("📁 Loading prepared data...")

    required_files = [
        "X_train.npy",
        "X_val.npy",
        "y_train.npy",
        "y_val.npy",
        "feature_names.pkl",
    ]

    for file in required_files:
        if not os.path.exists(os.path.join(DATA_DIR, file)):
            raise FileNotFoundError(f"Required file not found: {file}")

    # 피처명 로드
    with open(f"{DATA_DIR}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # 데이터 로드
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

    # Feature selection 적용
    if selected_features is not None:
        print(f"\n🎯 Applying feature selection...")

        # 선택된 피처가 데이터에 있는지 확인
        available_features = [f for f in selected_features if f in train_X.columns]
        missing_features = [f for f in selected_features if f not in train_X.columns]

        if missing_features:
            print(f"   ⚠️  Warning: {len(missing_features)} features not found in data")

        train_X = train_X[available_features]
        val_X = val_X[available_features]

        print(f"   ✓ Data filtered to {len(available_features)} features")

    print(f"   Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"   Features: {train_X.shape[1]}")

    return train_X, train_y, val_X, val_y


def train_kfold_xgboost(X, y, val_X, val_y, config, kfold_config):
    """Train XGBoost with K-Fold Cross Validation"""

    kf = StratifiedKFold(
        n_splits=kfold_config["n_splits"],
        shuffle=kfold_config["shuffle"],
        random_state=kfold_config["random_state"],
    )

    models = []
    fold_scores = []
    val_predictions = np.zeros(len(val_y))
    oof_predictions = np.zeros(len(y))

    print(f"\n🔄 Starting {kfold_config['n_splits']}-Fold Cross Validation...")

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n📊 Fold {fold + 1}/{kfold_config['n_splits']}")

        # Split data for this fold
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # Train model
        print(f"   Training fold {fold + 1}...")
        model = xgb.train(
            config,
            dtrain,
            num_boost_round=config["n_estimators"],
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=config.get("early_stopping_rounds", 30),
            verbose_eval=50,
        )

        models.append(model)

        # Predict on validation fold (OOF)
        fold_pred = model.predict(dvalid)
        oof_predictions[valid_idx] = fold_pred

        # Predict on test set
        dval = xgb.DMatrix(val_X)
        val_predictions += model.predict(dval) / kfold_config["n_splits"]

        # Calculate fold metrics
        fold_auc = roc_auc_score(y_valid_fold, fold_pred)
        fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
        fold_comp_score = fold_metrics["comp_score"]

        fold_scores.append(
            {"fold": fold + 1, "auc": fold_auc, "comp_score": fold_comp_score}
        )

        print(f"   ✓ Fold {fold + 1} AUC: {fold_auc:.6f}")
        print(f"   ✓ Fold {fold + 1} Competition Score: {fold_comp_score:.6f}")

        # WandB logging
        wandb.log({
            "fold_auc": fold_auc,
            "fold_comp_score": fold_comp_score,
            "best_iteration": model.best_iteration,
            "completed_folds": fold + 1,
            "fold_number": fold + 1,
        })

    return models, oof_predictions, val_predictions, fold_scores


def main():
    parser = argparse.ArgumentParser(description="XGBoost Training with Feature Selection")
    parser.add_argument("--n_features", type=int, default=None,
                       help="Number of top features to use")
    parser.add_argument("--keep_ratio", type=float, default=None,
                       help="Ratio of features to keep (0.0-1.0)")
    parser.add_argument("--importance_csv", type=str, default=None,
                       help="Path to feature importance CSV (auto-detect if not provided)")

    args = parser.parse_args()

    start_time = time.time()

    print("🚀 XGBoost K-Fold Training with Feature Selection")
    print("=" * 70)

    # Feature importance 파일 찾기
    if args.importance_csv is None:
        # 최신 체크포인트에서 자동으로 찾기
        checkpoint_base = CHECKPOINT_DIR
        all_checkpoints = [d for d in os.listdir(checkpoint_base)
                          if os.path.isdir(os.path.join(checkpoint_base, d))
                          and d.startswith('xgboost-')]
        all_checkpoints.sort(reverse=True)

        # feature_analysis 폴더가 있는 첫 번째 체크포인트 찾기
        importance_csv = None
        for checkpoint in all_checkpoints:
            candidate_path = os.path.join(checkpoint_base, checkpoint,
                                         "feature_analysis", "feature_importance.csv")
            if os.path.exists(candidate_path):
                importance_csv = candidate_path
                print(f"📊 Auto-detected importance file: {checkpoint}")
                break

        if importance_csv is None:
            print("❌ No feature importance file found!")
            print("   Run 'python -m src.xgboost.analyze_feature_importance' first")
            return
    else:
        importance_csv = args.importance_csv

    # Feature importance 로드
    importance_df = load_feature_importance(importance_csv)

    # Feature selection
    selected_features, importance_retained = select_top_features(
        importance_df,
        n_features=args.n_features,
        keep_ratio=args.keep_ratio
    )

    # Load data with feature selection
    try:
        data_start = time.time()
        train_X, train_y, val_X, val_y = load_prepared_data(selected_features)
        data_time = time.time() - data_start
        print(f"✓ Data loaded in {data_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        return

    # WandB 초기화
    timestamp = datetime.now().strftime("%m%d_%H%M")
    n_selected = len(selected_features)
    run_name = f"xgboost-kfold-selected-{n_selected}f-{timestamp}"

    wandb.init(
        project=XGBOOST_WANDB_CONFIG["PROJECT"],
        entity=XGBOOST_WANDB_CONFIG["ENTITY"],
        name=run_name,
        tags=XGBOOST_WANDB_CONFIG["TAGS"] + ["kfold", "feature-selection"],
        notes=f"XGBoost K-Fold with {n_selected} selected features ({importance_retained:.1f}% importance)",
        config={
            **XGBOOST_CONFIG,
            **KFOLD_CONFIG,
            "n_selected_features": n_selected,
            "total_features": len(importance_df),
            "feature_retention_ratio": n_selected / len(importance_df),
            "importance_retained_pct": importance_retained,
            "data_load_time": data_time,
            "num_features": train_X.shape[1],
            "train_samples": len(train_X),
            "val_samples": len(val_X),
            "train_ctr": train_y.mean(),
            "val_ctr": val_y.mean(),
        },
    )

    # Train K-Fold models
    train_start = time.time()
    models, oof_predictions, val_predictions, fold_scores = train_kfold_xgboost(
        train_X, train_y, val_X, val_y, XGBOOST_CONFIG, KFOLD_CONFIG
    )
    train_time = time.time() - train_start

    print("\n📊 Evaluating K-Fold results...")

    # Calculate metrics
    oof_comp_metrics = calculate_competition_metrics(train_y, oof_predictions)
    oof_metrics = {
        "oof_auc": oof_comp_metrics["auc"],
        "oof_ap": oof_comp_metrics["ap"],
        "oof_comp_score": oof_comp_metrics["comp_score"],
        "oof_wll": oof_comp_metrics["wll"],
    }

    val_comp_metrics = calculate_competition_metrics(val_y, val_predictions)
    val_metrics = {
        "val_auc": val_comp_metrics["auc"],
        "val_ap": val_comp_metrics["ap"],
        "val_comp_score": val_comp_metrics["comp_score"],
        "val_wll": val_comp_metrics["wll"],
    }

    # Print results
    print(f"\n📋 Individual Fold Results:")
    for fold_result in fold_scores:
        print(f"   Fold {fold_result['fold']}: AUC={fold_result['auc']:.6f}, "
              f"CompScore={fold_result['comp_score']:.6f}")

    fold_aucs = [f["auc"] for f in fold_scores]
    fold_comp_scores = [f["comp_score"] for f in fold_scores]

    print(f"\n📈 Cross-Validation Results:")
    print(f"   CV AUC: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
    print(f"   CV Competition Score: {np.mean(fold_comp_scores):.6f} ± {np.std(fold_comp_scores):.6f}")

    print(f"\n🎯 Out-of-Fold Metrics:")
    print(f"   OOF AUC: {oof_metrics['oof_auc']:.6f}")
    print(f"   OOF Competition Score: {oof_metrics['oof_comp_score']:.6f}")

    print(f"\n✨ Validation Metrics (Ensemble):")
    print(f"   Val AUC: {val_metrics['val_auc']:.6f}")
    print(f"   Val Competition Score: {val_metrics['val_comp_score']:.6f}")

    # Save models
    model_dir = f"{CHECKPOINT_DIR}/{run_name}"
    os.makedirs(model_dir, exist_ok=True)

    for i, model in enumerate(models):
        model.save_model(f"{model_dir}/xgboost_fold_{i+1}.json")

    # Save predictions
    pd.DataFrame({"oof_predictions": oof_predictions}).to_parquet(
        f"{model_dir}/oof_predictions.parquet"
    )
    pd.DataFrame({"val_predictions": val_predictions}).to_parquet(
        f"{model_dir}/val_predictions.parquet"
    )

    # Save selected features list
    with open(f"{model_dir}/selected_features.txt", "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    # WandB logging
    wandb.log({
        "final_val_comp_score": val_metrics["val_comp_score"],
        "final_val_auc": val_metrics["val_auc"],
        "final_cv_comp_score": np.mean(fold_comp_scores),
        "final_cv_auc": np.mean(fold_aucs),
        "cv_std": np.std(fold_comp_scores),
        "training_time_minutes": train_time / 60,
        "total_time_minutes": (time.time() - start_time) / 60,
        "target_achieved": val_metrics["val_comp_score"] >= 0.35,
    })

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("🎉 XGBoost K-Fold Training with Feature Selection Completed!")
    print("=" * 70)
    print(f"📊 Feature Selection Summary:")
    print(f"   Selected: {n_selected}/{len(importance_df)} features ({importance_retained:.1f}% importance)")
    print(f"\n📈 Performance:")
    print(f"   CV Competition Score: {np.mean(fold_comp_scores):.6f} ± {np.std(fold_comp_scores):.6f}")
    print(f"   Val Competition Score: {val_metrics['val_comp_score']:.6f}")
    print(f"\n⏱️  Timing:")
    print(f"   Data Load: {data_time:.2f}s")
    print(f"   Training: {train_time:.2f}s")
    print(f"   Total: {total_time:.2f}s")
    print(f"\n📁 Models saved: {model_dir}")

    # Target check
    target_score = 0.35
    achieved = val_metrics["val_comp_score"] >= target_score
    print(f"\n🎯 Target: {target_score}+ (current: {val_metrics['val_comp_score']:.6f})")
    if achieved:
        print("   ✅ TARGET ACHIEVED with feature selection!")

    wandb.finish()


if __name__ == "__main__":
    main()
