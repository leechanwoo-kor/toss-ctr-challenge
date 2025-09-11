import os
import time
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from optuna.integration.wandb import WeightsAndBiasesCallback

# 기존 스크립트에서 유틸리티 함수 가져오기
from train import calculate_competition_score

# --- 설정 ---
# 스크립트의 위치를 기준으로 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'

N_TRIALS = 50  # 시도할 튜닝 횟수 (필요에 따라 조절)
RANDOM_STATE = 42

# --- Optuna 목적 함수 ---
def objective(trial, X, y):
    """Optuna가 최적화할 목적 함수"""
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    param = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'verbosity': 0,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        param, 
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    y_pred = model.predict(dval)
    score, _, _ = calculate_competition_score(y_val, y_pred)

    return score

# --- 메인 튜닝 로직 ---
def main():
    """메인 튜닝 파이프라인"""
    start_time = time.time()
    print(f"🚀 Starting hyperparameter tuning with Optuna for {N_TRIALS} trials...")

    X = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="competition_score",
        wandb_kwargs={"project": "toss-ctr-challenge", "job_type": "tune"}
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_tuning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    )
    
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, callbacks=[wandb_callback])

    print("\n🏁 Tuning Finished!")
    print(f"  Number of finished trials: {len(study.trials)}")
    print("  Best trial:")
    trial = study.best_trial
    print(f"    Value (Competition Score): {trial.value:.6f}")
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    MODELS_DIR.mkdir(exist_ok=True)
    best_params_path = MODELS_DIR / 'best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=2)
    print(f"\n💾 Best hyperparameters saved to: {best_params_path}")

    print(f"\n✅ Tuning pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
