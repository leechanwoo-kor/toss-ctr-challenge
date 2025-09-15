import os
import gc
import time
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
import pickle
import json
import wandb

import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

from utils import print_memory, clear_memory, calculate_competition_score

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'

# 명명 규칙을 위한 설정
MODEL_TYPE = "xgb"
FEATURE_VERSION = "v2"
RUN_TYPE = "train"

N_FOLDS = 5
RANDOM_STATE = 42

# --- 피처 중요도 분석 함수 ---
def log_feature_importance(model, run_name):
    print("\nAnalyzing and logging feature importance...")
    importance_gain = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance_gain.keys()),
        'importance': list(importance_gain.values())
    }).sort_values('importance', ascending=False)

    csv_path = MODEL_DIR / f'{run_name}_feature_importance.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"Feature importance saved to: {csv_path}")
    wandb.log({"feature_importance_table": wandb.Table(dataframe=importance_df)})

    plt.figure(figsize=(12, 8))
    # FutureWarning 해결: `y`를 `hue`로 전달하고 legend=False 설정
    sns.barplot(x='importance', y='feature', data=importance_df.head(25), palette='viridis', hue='feature', legend=False)
    plt.title('Top 25 Feature Importance (Gain)')
    plt.tight_layout()
    
    plot_path = MODEL_DIR / f'{run_name}_feature_importance.png'
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    wandb.log({"feature_importance_plot": wandb.Image(str(plot_path))})
    plt.close()
    
    return csv_path, plot_path

# --- 메인 학습 로직 ---
def main():
    start_time = time.time()
    print("Starting training pipeline...")
    MODEL_DIR.mkdir(exist_ok=True)
    print_memory("Initial")

    print("\nLoading preprocessed data...")
    X_df = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded features shape: {X_df.shape}")
    print(f"Loaded target shape: {y.shape}")
    print_memory("After loading data")

    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    xgb_params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'verbosity': 0,
        'seed': RANDOM_STATE,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    best_params_path = MODEL_DIR / 'best_hyperparameters.json'
    if best_params_path.exists():
        print(f"\nLoading tuned hyperparameters from {best_params_path}...")
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        n_estimators = best_params.pop('n_estimators', 10)
        xgb_params.update(best_params)
        print("Updated XGBoost parameters with tuned values.")
    else:
        print("\nNo hyperparameter file found. Training with baseline parameters.")
        n_estimators = 10 # 기본 n_estimators 상향

    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"{MODEL_TYPE}-{FEATURE_VERSION}-{RUN_TYPE}-{timestamp}"

    print(f"\nRun Name: {run_name}")
    print(f"Final XGBoost Parameters: {json.dumps(xgb_params, indent=2)}")
    print(f"Max n_estimators: {n_estimators}")

    run = wandb.init(
        project="toss-ctr-challenge",
        name=run_name,
        config=xgb_params,
        job_type=RUN_TYPE,
        tags=[MODEL_TYPE, FEATURE_VERSION]
    )
    wandb.config.update({
        "n_folds": N_FOLDS,
        "max_n_estimators": n_estimators
    })

    try:
        print("\nStarting 5-Fold Cross-Validation...")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        cv_scores, cv_ap, cv_wll, cv_models = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), 1):
            fold_start = time.time()
            print(f"--- Fold {fold}/{N_FOLDS} ---")
            
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            y_pred = model.predict(dval)
            score, ap, wll = calculate_competition_score(y_val, y_pred)
            
            cv_scores.append(score)
            cv_ap.append(ap)
            cv_wll.append(wll)
            cv_models.append(model)
            
            print(f"  Score: {score:.6f} | AP: {ap:.6f} | WLL: {wll:.6f} | Best iter: {model.best_iteration}")
            print(f"  Fold time: {time.time() - fold_start:.1f}s")
            
            wandb.log({
                "fold": fold,
                "competition_score": score,
                "average_precision": ap,
                "weighted_logloss": wll,
                "best_iteration": model.best_iteration
            })

            del dtrain, dval, X_train, X_val, y_train, y_val
            clear_memory()

        print("\nCross-Validation Finished!")
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"Final CV Score: {mean_score:.6f} +-{std_score:.6f}")

        wandb.summary["cv_score_mean"] = mean_score
        wandb.summary["cv_score_std"] = std_score

        best_fold_idx = np.argmax(cv_scores)
        best_model = cv_models[best_fold_idx]
        
        print(f"\nSaving best model (Fold {best_fold_idx + 1}, Score: {cv_scores[best_fold_idx]:.6f})")
        
        model_path = MODEL_DIR / f'{run_name}.json'
        best_model.save_model(str(model_path))
        print(f"Model saved to: {model_path}")
        
        csv_path, plot_path = log_feature_importance(best_model, run_name)

        results = {
            'cv_score_mean': float(mean_score),
            'cv_score_std': float(std_score),
            'cv_scores': [float(s) for s in cv_scores],
            'best_fold': int(best_fold_idx + 1),
            'xgb_params': {k: v for k, v in xgb_params.items() if k != 'device'}
        }
        results_path = MODEL_DIR / f'{run_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(str(model_path))
        artifact.add_file(str(csv_path))
        artifact.add_file(str(plot_path))
        run.log_artifact(artifact)

    finally:
        run.finish()
        print(f"\nTraining pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
