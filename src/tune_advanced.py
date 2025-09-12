import os
import time
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import wandb

import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from optuna.integration.wandb import WeightsAndBiasesCallback

# 기존 스크립트에서 유틸리티 함수 가져오기
from train import calculate_competition_score

# --- 설정 ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

INPUT_X = PROCESSED_DIR / 'X_processed.parquet'
INPUT_Y = PROCESSED_DIR / 'y_processed.npy'

N_TRIALS = 200  # 대폭 증가
RANDOM_STATE = 42
N_FOLDS = 3  # 튜닝 시에는 빠른 평가를 위해 3-fold 사용

# --- 확장된 Optuna 목적 함수 ---
def objective(trial, X, y):
    """확장된 하이퍼파라미터 튜닝을 위한 목적 함수"""
    
    # 더 정교한 validation을 위해 3-fold CV 사용
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    # 확장된 하이퍼파라미터 공간
    param = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'device': 'cuda',
        'verbosity': 0,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'seed': RANDOM_STATE,
        
        # 기존 파라미터들 (범위 확장)
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        
        # 새로 추가된 파라미터들
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
    }
    
    # 동적 n_estimators 선택
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    early_stopping_rounds = max(20, n_estimators // 10)
    
    # 3-fold CV로 성능 평가
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
        
        # 중간 점수 보고 (pruning을 위해)
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(cv_scores)

# --- 메인 튜닝 로직 ---
def main():
    """확장된 하이퍼파라미터 튜닝 파이프라인"""
    start_time = time.time()
    print(f"🚀 Starting ADVANCED hyperparameter tuning with {N_TRIALS} trials...")
    print(f"Using {N_FOLDS}-fold CV for faster evaluation")

    X = pd.read_parquet(INPUT_X)
    y = np.load(INPUT_Y)
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    # wandb 초기화
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="competition_score",
        wandb_kwargs={
            "project": "toss-ctr-challenge", 
            "job_type": "advanced_tune",
            "tags": ["advanced_tuning", "200_trials", "extended_params"]
        }
    )

    # Optuna study 생성 (pruning 활성화)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_advanced_tuning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y), 
        n_trials=N_TRIALS, 
        callbacks=[wandb_callback],
        show_progress_bar=True
    )

    print("\n🏁 Advanced Tuning Finished!")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print("  Best trial:")
    trial = study.best_trial
    print(f"    Value (Competition Score): {trial.value:.6f}")
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

    # 최적 파라미터 저장
    MODELS_DIR.mkdir(exist_ok=True)
    best_params_path = MODELS_DIR / 'best_hyperparameters_advanced.json'
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=2)
    print(f"\n💾 Advanced best hyperparameters saved to: {best_params_path}")

    # 이전 베스트와 비교
    old_best_path = MODELS_DIR / 'best_hyperparameters.json'
    if old_best_path.exists():
        print(f"\n📊 Comparison with previous best:")
        print(f"  Previous CV score (estimated): ~0.3507")
        print(f"  New CV score: {trial.value:.6f}")
        improvement = trial.value - 0.3507
        print(f"  Improvement: {improvement:+.6f}")

    print(f"\n✅ Advanced tuning pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()