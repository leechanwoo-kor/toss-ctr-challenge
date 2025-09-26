"""
CatBoost with K-Fold Cross Validation for CTR Prediction
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import wandb
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import CATBOOST_CONFIG, CATBOOST_WANDB_CONFIG, KFOLD_CONFIG
from src.utils.metrics import calculate_competition_metrics

# WandB 설정 업데이트
CATBOOST_WANDB_CONFIG_RUN = {
    'project': CATBOOST_WANDB_CONFIG['PROJECT'],
    'entity': 'leechanwoo',
    'name': f'catboost-kfold-{datetime.now().strftime("%m%d_%H%M")}',
    'tags': CATBOOST_WANDB_CONFIG['TAGS']
}


def load_prepared_data():
    """Load prepared training and validation data (XGBoost와 동일한 데이터 사용)"""
    print("📁 Loading prepared data...")

    data_dir = 'data/processed/xgboost'  # XGBoost와 동일한 데이터 사용

    # 데이터 파일 존재 확인
    required_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy',
                      'metadata.pkl', 'processor.pkl', 'feature_names.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Required file not found: {file}. Run 'python -m src.xgboost.xgboost_prepare_data' first.")

    try:
        # 피처명 먼저 로드
        with open(f'{data_dir}/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        # 데이터를 바로 DataFrame으로 로드 (메모리 효율적)
        train_X = pd.DataFrame(
            np.load(f'{data_dir}/X_train.npy'), columns=feature_names
        )
        val_X = pd.DataFrame(
            np.load(f'{data_dir}/X_val.npy'), columns=feature_names
        )
        train_y = pd.Series(np.load(f'{data_dir}/y_train.npy'))
        val_y = pd.Series(np.load(f'{data_dir}/y_val.npy'))

        # 프로세서 로드
        with open(f'{data_dir}/processor.pkl', 'rb') as f:
            processor = pickle.load(f)

    except Exception as e:
        raise RuntimeError(f"Failed to load prepared data: {e}")

    print(f"Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"Features: {len(feature_names)}")

    return train_X, train_y, val_X, val_y, feature_names, processor


def get_categorical_features(feature_names, processor):
    """Get categorical feature indices for CatBoost (프로세서에서 자동 감지)"""
    categorical_features = []
    categorical_feature_names = processor.categorical_features

    for i, name in enumerate(feature_names):
        if name in categorical_feature_names:
            categorical_features.append(i)

    print(f"Categorical features found: {[feature_names[i] for i in categorical_features]}")
    return categorical_features


def train_catboost_kfold(X, y, val_X, val_y, feature_names, categorical_features):
    """Train CatBoost with K-Fold cross validation"""
    print("🔄 Starting 5-Fold Cross Validation...")

    # K-fold setup (stratify 파라미터 제거)
    kfold = StratifiedKFold(
        n_splits=KFOLD_CONFIG['n_splits'],
        shuffle=KFOLD_CONFIG['shuffle'],
        random_state=KFOLD_CONFIG['random_state']
    )

    # Convert categorical features to string for CatBoost
    categorical_feature_names = [feature_names[i] for i in categorical_features]
    print(f"Converting categorical features to string: {categorical_feature_names}")

    X_cat = X.copy()
    val_X_cat = val_X.copy()

    for i in categorical_features:
        col_name = feature_names[i]
        X_cat[col_name] = X_cat[col_name].astype(str)
        val_X_cat[col_name] = val_X_cat[col_name].astype(str)

    # Results storage
    fold_results = []
    oof_predictions = np.zeros(len(X))
    val_predictions = np.zeros(len(val_X))
    models = []

    # Train each fold
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_cat, y)):
        print(f"\n📊 Fold {fold+1}/{KFOLD_CONFIG['n_splits']}")
        print(f"🚀 Training fold {fold+1}...")

        # Split data (use categorical converted data)
        X_train_fold = X_cat.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X_cat.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        # Create CatBoost pools
        train_pool = Pool(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_features
        )

        valid_pool = Pool(
            X_valid_fold,
            y_valid_fold,
            cat_features=categorical_features
        )

        # Train model
        model = CatBoostClassifier(**CATBOOST_CONFIG)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=CATBOOST_CONFIG['verbose']
        )

        # Make predictions
        fold_pred = model.predict_proba(X_valid_fold)[:, 1]
        val_pred = model.predict_proba(val_X_cat)[:, 1]

        # Store results
        oof_predictions[valid_idx] = fold_pred
        val_predictions += val_pred / KFOLD_CONFIG['n_splits']
        models.append(model)

        # Calculate metrics
        fold_auc = roc_auc_score(y_valid_fold, fold_pred)
        fold_comp_metrics = calculate_competition_metrics(y_valid_fold.values, fold_pred)

        # Handle both dict and float return types
        if isinstance(fold_comp_metrics, dict):
            fold_comp_score = fold_comp_metrics.get('comp_score', fold_comp_metrics.get('competition_score', fold_comp_metrics.get('score', 0)))
        else:
            fold_comp_score = fold_comp_metrics

        fold_results.append({
            'fold': fold + 1,
            'auc': fold_auc,
            'comp_score': fold_comp_score
        })

        print(f"   Fold {fold+1} AUC: {fold_auc:.6f}")
        print(f"   Fold {fold+1} Competition Score: {fold_comp_score:.6f}")

        # Log to wandb
        wandb.log({
            f'fold_{fold+1}_auc': fold_auc,
            f'fold_{fold+1}_comp_score': fold_comp_score,
            f'fold_{fold+1}_best_iteration': model.get_best_iteration(),
            'completed_folds': fold + 1
        })

    return fold_results, oof_predictions, val_predictions, models


def evaluate_kfold_results(y_true, oof_predictions, val_y, val_predictions, fold_results):
    """Evaluate K-Fold results"""
    print("🔮 Evaluating K-Fold results...")

    # Individual fold results
    print("\n📊 Individual Fold Results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: AUC={result['auc']:.6f}, CompScore={result['comp_score']:.6f}")

    # Cross-validation metrics
    fold_aucs = [r['auc'] for r in fold_results]
    fold_comp_scores = [r['comp_score'] for r in fold_results]

    cv_auc_mean = np.mean(fold_aucs)
    cv_auc_std = np.std(fold_aucs)
    cv_comp_mean = np.mean(fold_comp_scores)
    cv_comp_std = np.std(fold_comp_scores)

    print(f"\n📈 Cross-Validation Results:")
    print(f"  CV AUC: {cv_auc_mean:.6f} ± {cv_auc_std:.6f}")
    print(f"  CV Competition Score: {cv_comp_mean:.6f} ± {cv_comp_std:.6f}")

    # Out-of-fold metrics
    oof_auc = roc_auc_score(y_true, oof_predictions)
    oof_comp_metrics = calculate_competition_metrics(y_true.values, oof_predictions)

    # Handle both dict and float return types
    if isinstance(oof_comp_metrics, dict):
        oof_comp_score = oof_comp_metrics.get('comp_score', oof_comp_metrics.get('competition_score', oof_comp_metrics.get('score', 0)))
    else:
        oof_comp_score = oof_comp_metrics

    print(f"\n🎯 Out-of-Fold Metrics:")
    print(f"  OOF AUC: {oof_auc:.6f}")
    print(f"  OOF Competition Score: {oof_comp_score:.6f}")

    # Validation metrics
    val_auc = roc_auc_score(val_y, val_predictions)
    val_comp_metrics = calculate_competition_metrics(val_y.values, val_predictions)

    # Handle both dict and float return types
    if isinstance(val_comp_metrics, dict):
        val_comp_score = val_comp_metrics.get('comp_score', val_comp_metrics.get('competition_score', val_comp_metrics.get('score', 0)))
    else:
        val_comp_score = val_comp_metrics

    print(f"\n🎯 Validation Metrics (Ensemble):")
    print(f"  Val AUC: {val_auc:.6f}")
    print(f"  Val Competition Score: {val_comp_score:.6f}")

    # Log final metrics to wandb
    wandb.log({
        'cv_auc_mean': cv_auc_mean,
        'cv_auc_std': cv_auc_std,
        'cv_comp_score_mean': cv_comp_mean,
        'cv_comp_score_std': cv_comp_std,
        'oof_auc': oof_auc,
        'oof_comp_score': oof_comp_score,
        'val_auc': val_auc,
        'val_comp_score': val_comp_score
    })

    return {
        'cv_auc_mean': cv_auc_mean,
        'cv_comp_mean': cv_comp_mean,
        'oof_auc': oof_auc,
        'oof_comp_score': oof_comp_score,
        'val_auc': val_auc,
        'val_comp_score': val_comp_score
    }


def save_models_and_predictions(models, oof_predictions, val_predictions, feature_names, run_name):
    """Save trained models and predictions"""
    save_dir = f'checkpoints/catboost/{run_name}'
    os.makedirs(save_dir, exist_ok=True)

    # Save models
    for i, model in enumerate(models):
        model.save_model(f'{save_dir}/catboost_fold_{i+1}.cbm')

    # Save predictions
    np.save(f'{save_dir}/oof_predictions.npy', oof_predictions)
    np.save(f'{save_dir}/val_predictions.npy', val_predictions)

    # Save feature names
    with open(f'{save_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    print(f"Models saved: {save_dir}")
    return save_dir


def main():
    """Main training function"""
    print("🚀 Starting CatBoost K-Fold training...")

    start_time = time.time()

    # Load data
    load_start = time.time()
    X, y, val_X, val_y, feature_names, processor = load_prepared_data()
    load_time = time.time() - load_start
    print(f"Data loaded in {load_time:.2f} seconds")

    # Get categorical features from processor
    categorical_features = get_categorical_features(feature_names, processor)
    print(f"Categorical features: {categorical_features}")

    # Initialize wandb
    wandb.init(**CATBOOST_WANDB_CONFIG_RUN)
    wandb.config.update(CATBOOST_CONFIG)
    wandb.config.update(KFOLD_CONFIG)
    wandb.config.update({
        'train_shape': X.shape,
        'val_shape': val_X.shape,
        'n_features': len(feature_names),
        'n_categorical': len(categorical_features)
    })

    # Train models
    train_start = time.time()
    fold_results, oof_predictions, val_predictions, models = train_catboost_kfold(
        X, y, val_X, val_y, feature_names, categorical_features
    )
    train_time = time.time() - train_start

    # Evaluate results
    final_metrics = evaluate_kfold_results(y, oof_predictions, val_y, val_predictions, fold_results)

    # Save models
    run_name = CATBOOST_WANDB_CONFIG_RUN['name']
    save_dir = save_models_and_predictions(models, oof_predictions, val_predictions, feature_names, run_name)

    total_time = time.time() - start_time

    # Final summary
    print("\n" + "="*50)
    print("🎉 CatBoost K-Fold Training Completed!")
    print("="*50)
    print(f"Data Load Time: {load_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Models saved: {save_dir}")
    print(f"🎯 Target: 0.35+ (current: {final_metrics['val_comp_score']:.6f})")

    if final_metrics['val_comp_score'] >= 0.35:
        print("🎉 TARGET ACHIEVED with K-Fold ensemble!")
    else:
        print("🔄 Continue optimization to reach target...")

    wandb.finish()

    return final_metrics


if __name__ == "__main__":
    final_metrics = main()