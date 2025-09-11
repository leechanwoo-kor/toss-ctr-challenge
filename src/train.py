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

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'

N_FOLDS = 5
RANDOM_STATE = 42

# --- 유틸리티 함수 ---
def print_memory(context=""):
    mem = psutil.virtual_memory()
    print(f"💾 {context} Memory: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")

def clear_memory():
    gc.collect()
    print("🧹 Memory cleared")

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
    print("🚀 Starting training pipeline...")
    MODEL_DIR.mkdir(exist_ok=True)
    print_memory("Initial")

    print("\n📦 Loading preprocessed data...")
    X_df = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded features shape: {X_df.shape}")
    print(f"Loaded target shape: {y.shape}")
    print_memory("After loading data")

    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    
    # 기본 XGBoost 파라미터
    xgb_params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'verbosity': 0,
        'seed': RANDOM_STATE,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        # 베이스라인 파라미터
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # 최적화된 하이퍼파라미터 로드 및 업데이트
    best_params_path = MODEL_DIR / 'best_hyperparameters.json'
    run_tags = ["xgboost"]
    if best_params_path.exists():
        print(f"\n튜닝된 최적 파라미터를 {best_params_path} 에서 불러옵니다.")
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        xgb_params.update(best_params)
        print("XGBoost 파라미터를 튜닝된 값으로 업데이트했습니다.")
        run_tags.append("tuned")
    else:
        print("\n최적 파라미터 파일이 없습니다. 기본 파라미터로 학습을 진행합니다.")
        run_tags.append("baseline")

    print(f"\n📊 Final XGBoost Parameters: {json.dumps(xgb_params, indent=2)}")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    run_name = f"{'-'.join(run_tags)}_{timestamp}"

    run = wandb.init(
        project="toss-ctr-challenge",
        name=run_name,
        config=xgb_params,
        job_type="train",
        tags=run_tags
    )
    wandb.config.update({"n_folds": N_FOLDS})

    try:
        print("\n🔄 Starting 5-Fold Cross-Validation...")
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
                num_boost_round=200,
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

        print("\n🏁 Cross-Validation Finished!")
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"🏆 Final CV Score: {mean_score:.6f} ± {std_score:.6f}")

        wandb.summary["cv_score_mean"] = mean_score
        wandb.summary["cv_score_std"] = std_score

        best_fold_idx = np.argmax(cv_scores)
        best_model = cv_models[best_fold_idx]
        best_score = cv_scores[best_fold_idx]
        
        print(f"\n💾 Saving best model (Fold {best_fold_idx + 1}, Score: {best_score:.6f})")
        
        model_path = MODEL_DIR / f'{run_name}_best_model.json'
        best_model.save_model(str(model_path))
        print(f"Model saved to: {model_path}")
        
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
        run.log_artifact(artifact)

    finally:
        run.finish()
        print(f"\n✅ Training pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()