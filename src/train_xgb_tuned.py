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

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'

INPUT_X = PROCESSED_DIR / 'X_engineered.parquet'
INPUT_Y = PROCESSED_DIR / 'y_engineered.npy'

N_FOLDS = 5
RANDOM_STATE = 42

# --- 유틸리티 함수 ---
def print_memory(context=""):
    mem = psutil.virtual_memory()
    print(f"Memory {context}: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")

def clear_memory():
    gc.collect()
    print("Memory cleared")

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0
    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

# --- 메인 학습 로직 ---
def main():
    start_time = time.time()
    print("Starting XGBoost training with TUNED hyperparameters...")
    MODEL_DIR.mkdir(exist_ok=True)
    print_memory("Initial")

    print("Loading engineered data...")
    X_df = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded features shape: {X_df.shape}")
    print(f"Loaded target shape: {y.shape}")
    print_memory("After loading data")

    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    # 튜닝에서 발견한 최적 파라미터 (Trial 2: CV 0.354387)
    xgb_params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'verbosity': 0,
        'seed': RANDOM_STATE,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        # 최적화된 하이퍼파라미터
        'max_depth': 10,
        'learning_rate': 0.012792877895237259,  # 매우 낮은 학습률
        'subsample': 0.9665792438277845,
        'colsample_bytree': 0.7660261892199351,
        'gamma': 0.04405067672110807,
        'min_child_weight': 4,
        'reg_alpha': 0.22176511915149436,
        'reg_lambda': 67.49582620763466,  # 높은 L2 정규화
        'max_delta_step': 4,
        'colsample_bylevel': 0.8144941575324838,
        'colsample_bynode': 0.6794417875304353,
    }

    print(f"Tuned XGBoost Parameters: {json.dumps(xgb_params, indent=2)}")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    run_name = f"xgboost-tuned_{timestamp}"

    run = wandb.init(
        project="toss-ctr-challenge",
        name=run_name,
        config=xgb_params,
        job_type="train",
        tags=["xgboost", "engineered_features", "tuned_hyperparams", "best_trial"]
    )
    wandb.config.update({
        "n_folds": N_FOLDS, 
        "n_features": X_df.shape[1],
        "tuning_trial": 2,
        "expected_cv_score": 0.354387
    })

    try:
        print("Starting 5-Fold Cross-Validation with tuned parameters...")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        cv_scores, cv_ap, cv_wll, cv_models = [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), 1):
            fold_start = time.time()
            print(f"--- Fold {fold}/{N_FOLDS} ---")
            
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # 튜닝에서 발견한 최적 n_estimators 사용
            model = xgb.train(
                xgb_params, 
                dtrain,
                num_boost_round=557,  # 튜닝에서 발견한 최적값
                evals=[(dval, 'val')],
                early_stopping_rounds=20,
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

        print("Cross-Validation Finished!")
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"Final CV Score: {mean_score:.6f} ± {std_score:.6f}")
        print(f"Expected score from tuning: 0.354387")
        print(f"Improvement vs baseline: {mean_score - 0.351796:.6f}")

        wandb.summary["cv_score_mean"] = mean_score
        wandb.summary["cv_score_std"] = std_score
        wandb.summary["improvement_vs_baseline"] = mean_score - 0.351796

        best_fold_idx = np.argmax(cv_scores)
        best_model = cv_models[best_fold_idx]
        best_score = cv_scores[best_fold_idx]
        
        print(f"Saving best model (Fold {best_fold_idx + 1}, Score: {best_score:.6f})")
        
        model_path = MODEL_DIR / f'{run_name}_best_model.json'
        best_model.save_model(str(model_path))
        print(f"Model saved to: {model_path}")
        
        results = {
            'cv_score_mean': float(mean_score),
            'cv_score_std': float(std_score),
            'cv_scores': [float(s) for s in cv_scores],
            'best_fold': int(best_fold_idx + 1),
            'xgb_params': {k: v for k, v in xgb_params.items() if k != 'device'},
            'tuning_info': {
                'trial_number': 2,
                'expected_score': 0.354387,
                'baseline_score': 0.351796
            }
        }
        results_path = MODEL_DIR / f'{run_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(str(model_path))
        run.log_artifact(artifact)

    finally:
        run.finish()
        print(f"Tuned XGBoost training pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()