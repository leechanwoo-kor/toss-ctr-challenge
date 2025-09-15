import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import optuna
from sklearn.model_selection import StratifiedKFold
from optuna.integration.wandb import WeightsAndBiasesCallback
import xgboost as xgb

from utils import calculate_competition_score

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'

# 명명 규칙 및 튜닝 설정
MODEL_TYPE = "xgb"
FEATURE_VERSION = "v2"
RUN_TYPE = "tune"

N_TRIALS = 20
RANDOM_STATE = 42
N_FOLDS = 3

# --- Optuna 목적 함수 ---
def objective(trial, X, y):
    """Optuna가 최적화할 목적 함수 (CV, Pruning 적용)"""
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

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
    
    # n_estimators는 튜닝 대상에서 제외하고, early stopping에 맡김
    n_estimators = 3000
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
        
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_scores)

# --- 메인 튜닝 로직 ---
def main():
    """메인 튜닝 파이프라인"""
    start_time = time.time()
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"{MODEL_TYPE}-{FEATURE_VERSION}-{RUN_TYPE}-{timestamp}"
    
    print(f"Starting hyperparameter tuning with Optuna for {N_TRIALS} trials...")
    print(f"Run Name: {run_name}")

    X = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    wandb_callback = WeightsAndBiasesCallback(
        metric_name="competition_score",
        wandb_kwargs={
            "project": "toss-ctr-challenge", 
            "name": run_name,
            "job_type": RUN_TYPE,
            "tags": [MODEL_TYPE, FEATURE_VERSION, f"{N_TRIALS}_trials"]
        }
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=run_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y), 
        n_trials=N_TRIALS, 
        callbacks=[wandb_callback],
        show_progress_bar=True
    )

    print("\nTuning Finished!")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print("  Best trial:")
    trial = study.best_trial
    print(f"    Value (Mean CV Score): {trial.value:.6f}")
    print("    Params: ")
    best_params = trial.params.copy()
    for key, value in best_params.items():
        print(f"      {key}: {value}")

    MODELS_DIR.mkdir(exist_ok=True)
    best_params_path = MODELS_DIR / 'best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters saved to: {best_params_path}")

    print(f"\nTuning pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
