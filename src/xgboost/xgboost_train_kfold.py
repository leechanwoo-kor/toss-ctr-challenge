"""
XGBoost with K-Fold Cross Validation for CTR Prediction
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import wandb
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import XGBOOST_CONFIG, XGBOOST_WANDB_CONFIG, KFOLD_CONFIG
from src.utils.metrics import calculate_competition_metrics


def load_prepared_data():
    """Load prepared training and validation data"""
    print("📁 Loading prepared data...")

    data_dir = 'data/processed/xgboost_enhanced'

    # 데이터 파일 존재 확인
    required_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy',
                      'metadata.pkl', 'processor.pkl', 'feature_names.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Required file not found: {file}. Run 'python -m src.xgboost.xgboost_prepare_data' first.")

    # 데이터 로드
    X_train = np.load(f'{data_dir}/X_train.npy')
    X_val = np.load(f'{data_dir}/X_val.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_val = np.load(f'{data_dir}/y_val.npy')

    # 피처명 로드
    with open(f'{data_dir}/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # pandas DataFrame으로 변환 (K-fold를 위해)
    train_X = pd.DataFrame(X_train, columns=feature_names)
    train_y = pd.Series(y_train)
    val_X = pd.DataFrame(X_val, columns=feature_names)
    val_y = pd.Series(y_val)

    print(f"Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"Features: {len(feature_names)}")

    return train_X, train_y, val_X, val_y


def train_kfold_xgboost(X, y, val_X, val_y, config, kfold_config):
    """Train XGBoost with K-Fold Cross Validation"""

    # Initialize K-Fold
    kf = StratifiedKFold(
        n_splits=kfold_config["n_splits"],
        shuffle=kfold_config["shuffle"],
        random_state=kfold_config["random_state"]
    )

    models = []
    fold_scores = []
    val_predictions = np.zeros(len(val_y))
    oof_predictions = np.zeros(len(y))

    print(f"🔄 Starting {kfold_config['n_splits']}-Fold Cross Validation...")

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

        # Train model for this fold
        print(f"🚀 Training fold {fold + 1}...")
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        # WandB 콜백 클래스 (fold별)
        class WandBFoldCallback(xgb.callback.TrainingCallback):
            """각 fold의 training 과정을 WandB에 로깅하는 콜백"""

            def __init__(self, fold_num):
                self.fold_num = fold_num

            def after_iteration(self, model, epoch, evals_log):
                """매 iteration 후 호출"""
                if epoch % 20 == 0:  # 20 iteration마다 로깅
                    log_dict = {
                        'fold': self.fold_num,
                        'iteration': epoch,
                        'fold_iteration': f"fold_{self.fold_num}_iter_{epoch}"
                    }

                    # evaluation 결과 로깅
                    for eval_name, metrics in evals_log.items():
                        for metric_name, values in metrics.items():
                            if values:  # 값이 있는 경우만
                                log_dict[f'fold_{self.fold_num}_{eval_name}_{metric_name}'] = values[-1]

                    wandb.log(log_dict)

                return False  # 학습을 계속하려면 False

        wandb_callback = WandBFoldCallback(fold + 1)

        model = xgb.train(
            config,
            dtrain,
            num_boost_round=config["n_estimators"],
            evals=watchlist,
            early_stopping_rounds=config["early_stopping_rounds"],
            verbose_eval=50  # 50 iteration마다 출력
        )

        models.append(model)

        # Predict on validation fold (out-of-fold)
        dvalid_pred = xgb.DMatrix(X_valid_fold)
        oof_predictions[valid_idx] = model.predict(dvalid_pred)

        # Predict on test set
        dval = xgb.DMatrix(val_X)
        val_predictions += model.predict(dval) / kfold_config["n_splits"]

        # Calculate fold metrics
        fold_pred = model.predict(dvalid_pred)
        fold_auc = roc_auc_score(y_valid_fold, fold_pred)
        fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
        fold_comp_score = fold_metrics['comp_score']

        fold_scores.append({
            'fold': fold + 1,
            'auc': fold_auc,
            'comp_score': fold_comp_score
        })

        print(f"   Fold {fold + 1} AUC: {fold_auc:.6f}")
        print(f"   Fold {fold + 1} Competition Score: {fold_comp_score:.6f}")

        # 각 fold 완료 후 WandB에 로깅
        wandb.log({
            f'fold_{fold+1}_final_auc': fold_auc,
            f'fold_{fold+1}_final_comp_score': fold_comp_score,
            f'fold_{fold+1}_best_iteration': model.best_iteration,
            'completed_folds': fold + 1
        })

    return models, oof_predictions, val_predictions, fold_scores


def evaluate_predictions(y_true, y_pred, prefix=""):
    """Calculate evaluation metrics"""
    comp_metrics = calculate_competition_metrics(y_true, y_pred)

    metrics = {
        f"{prefix}auc": comp_metrics['auc'],
        f"{prefix}ap": comp_metrics['ap'],
        f"{prefix}comp_score": comp_metrics['comp_score'],
        f"{prefix}wll": comp_metrics['wll'],
    }

    return metrics


def main():
    start_time = time.time()

    print("🚀 Starting XGBoost K-Fold training...")

    # Load data
    data_start = time.time()
    train_X, train_y, val_X, val_y = load_prepared_data()
    data_time = time.time() - data_start
    print(f"Data loaded in {data_time:.2f} seconds")

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
            **XGBOOST_CONFIG,
            **KFOLD_CONFIG,
            "data_load_time": data_time,
            "num_features": train_X.shape[1],
            "train_samples": len(train_X),
            "val_samples": len(val_X),
            "train_ctr": train_y.mean(),
            "val_ctr": val_y.mean(),
        }
    )

    # Train K-Fold models
    train_start = time.time()
    models, oof_predictions, val_predictions, fold_scores = train_kfold_xgboost(
        train_X, train_y, val_X, val_y, XGBOOST_CONFIG, KFOLD_CONFIG
    )
    train_time = time.time() - train_start

    print("\n🔮 Evaluating K-Fold results...")

    # Calculate Out-of-Fold metrics
    oof_metrics = evaluate_predictions(train_y, oof_predictions, "oof_")

    # Calculate validation metrics
    val_metrics = evaluate_predictions(val_y, val_predictions, "val_")

    # Print fold results
    print(f"\n📊 Individual Fold Results:")
    for fold_result in fold_scores:
        print(f"  Fold {fold_result['fold']}: AUC={fold_result['auc']:.6f}, CompScore={fold_result['comp_score']:.6f}")

    # Calculate fold statistics
    fold_aucs = [f['auc'] for f in fold_scores]
    fold_comp_scores = [f['comp_score'] for f in fold_scores]

    print(f"\n📈 Cross-Validation Results:")
    print(f"  CV AUC: {np.mean(fold_aucs):.6f} ± {np.std(fold_aucs):.6f}")
    print(f"  CV Competition Score: {np.mean(fold_comp_scores):.6f} ± {np.std(fold_comp_scores):.6f}")

    print(f"\n🎯 Out-of-Fold Metrics:")
    print(f"  OOF AUC: {oof_metrics['oof_auc']:.6f}")
    print(f"  OOF Competition Score: {oof_metrics['oof_comp_score']:.6f}")

    print(f"\n🎯 Validation Metrics (Ensemble):")
    print(f"  Val AUC: {val_metrics['val_auc']:.6f}")
    print(f"  Val Competition Score: {val_metrics['val_comp_score']:.6f}")

    # Save models
    model_dir = f"checkpoints/xgboost/{run_name}"
    os.makedirs(model_dir, exist_ok=True)

    for i, model in enumerate(models):
        model.save_model(f"{model_dir}/xgboost_fold_{i+1}.json")

    # Save predictions separately
    pd.DataFrame({'oof_predictions': oof_predictions}).to_parquet(f"{model_dir}/oof_predictions.parquet")
    pd.DataFrame({'val_predictions': val_predictions}).to_parquet(f"{model_dir}/val_predictions.parquet")

    # Log to WandB
    wandb.log({
        "training_time": train_time,
        "total_time": time.time() - start_time,
        "cv_auc_mean": np.mean(fold_aucs),
        "cv_auc_std": np.std(fold_aucs),
        "cv_comp_score_mean": np.mean(fold_comp_scores),
        "cv_comp_score_std": np.std(fold_comp_scores),
        **oof_metrics,
        **val_metrics,
    })

    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("🎉 XGBoost K-Fold Training Completed!")
    print("="*50)
    print(f"Data Load Time: {data_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Models saved: {model_dir}")

    # Target achievement check
    target_score = 0.35
    achieved = val_metrics['val_comp_score'] >= target_score
    print(f"🎯 Target: {target_score}+ (current: {val_metrics['val_comp_score']:.6f})")
    if achieved:
        print("🎉 TARGET ACHIEVED with K-Fold ensemble!")

    wandb.finish()


if __name__ == "__main__":
    main()