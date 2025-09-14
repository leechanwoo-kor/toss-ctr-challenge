import os
import time
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from optuna.integration.wandb import WeightsAndBiasesCallback

# 중앙 유틸리티 함수 가져오기
from utils import calculate_competition_score

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'

N_TRIALS = 100  # 시도할 튜닝 횟수
RANDOM_STATE = 42
N_FOLDS = 3     # 튜닝 시에는 빠른 평가를 위해 3-fold 사용

# --- Optuna 목적 함수 ---
def objective(trial, X, y):
    """Optuna가 최적화할 목적 함수 (CV, Pruning 적용)"""
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    # 확장된 하이퍼파라미터 탐색 공간
    param = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'verbosity': 0,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'seed': RANDOM_STATE,
        
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    n_estimators = trial.suggest_int('n_estimators', 500, 3000)
    early_stopping_rounds = 50
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            param, 
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        y_pred = model.predict(dval)
        score, _, _ = calculate_competition_score(y_val, y_pred)
        cv_scores.append(score)
        
        # Pruning을 위한 중간 결과 보고
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_scores)

# --- 메인 튜닝 로직 ---
def main():
    """메인 튜닝 파이프라인"""
    start_time = time.time()
    print(f"🚀 Starting hyperparameter tuning with Optuna for {N_TRIALS} trials...")
    print(f"Using {N_FOLDS}-fold CV and Pruning for robust evaluation.")

    X = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="competition_score",
        wandb_kwargs={
            "project": "toss-ctr-challenge", 
            "job_type": "tune",
            "tags": ["optuna_cv", "pruning"]
        }
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_cv_tuning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y), 
        n_trials=N_TRIALS, 
        callbacks=[wandb_callback],
        show_progress_bar=True
    )

    print("\n🏁 Tuning Finished!")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print("  Best trial:")
    trial = study.best_trial
    print(f"    Value (Mean CV Score): {trial.value:.6f}")
    print("    Params: ")
    # n_estimators는 early stopping으로 최적값이 결정되므로, 최종 파라미터에 추가
    best_params = trial.params.copy()
    # 실제 학습 시에는 early stopping을 사용하므로, 튜닝된 n_estimators는 최대치로만 활용
    # best_params['n_estimators'] = trial.params['n_estimators'] 
    for key, value in best_params.items():
        print(f"      {key}: {value}")

    MODELS_DIR.mkdir(exist_ok=True)
    best_params_path = MODELS_DIR / 'best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n💾 Best hyperparameters saved to: {best_params_path}")

    print(f"\n✅ Tuning pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()