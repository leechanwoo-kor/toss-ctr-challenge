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
from config import (
    MODEL_TYPE, FEATURE_VERSION, RANDOM_STATE, N_FOLDS,
    OUTPUT_X, OUTPUT_Y, BEST_PARAMS_PATH, DEFAULT_N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS, get_base_xgb_params, get_model_paths,
    create_directories
)

# 훈련 전용 설정
RUN_TYPE = "train"

# --- 피처 중요도 분석 함수 ---
def log_feature_importance(model, paths):
    """피처 중요도 분석 및 저장 (구조화된 경로 사용)"""
    print("\nAnalyzing and logging feature importance...")
    importance_gain = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance_gain.keys()),
        'importance': list(importance_gain.values())
    }).sort_values('importance', ascending=False)

    # 구조화된 경로 사용
    csv_path = paths['feature_importance_csv']
    importance_df.to_csv(csv_path, index=False)
    print(f"Feature importance saved to: {csv_path}")
    wandb.log({"feature_importance_table": wandb.Table(dataframe=importance_df)})

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(25),
                palette='viridis', hue='feature', legend=False)
    plt.title('Top 25 Feature Importance (Gain)')
    plt.tight_layout()

    plot_path = paths['feature_importance_png']
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    wandb.log({"feature_importance_plot": wandb.Image(str(plot_path))})
    plt.close()

    return csv_path, plot_path

# --- 메인 학습 로직 ---
def main():
    start_time = time.time()
    print("Starting training pipeline...")

    # 구조화된 디렉토리 생성
    create_directories()
    print_memory("Initial")

    print("\nLoading preprocessed data...")
    X_df = pd.read_parquet(OUTPUT_X)
    y = np.load(OUTPUT_Y)
    print(f"Loaded features shape: {X_df.shape}")
    print(f"Loaded target shape: {y.shape}")
    print_memory("After loading data")

    pos_ratio = y.mean()

    # config에서 기본 파라미터 가져오기
    xgb_params = get_base_xgb_params(pos_ratio)

    if BEST_PARAMS_PATH.exists():
        print(f"\nLoading tuned hyperparameters from {BEST_PARAMS_PATH}...")
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)

        n_estimators = best_params.pop('n_estimators', DEFAULT_N_ESTIMATORS)
        xgb_params.update(best_params)
        print("Updated XGBoost parameters with tuned values.")
    else:
        print("\nNo hyperparameter file found. Training with baseline parameters.")
        n_estimators = DEFAULT_N_ESTIMATORS

    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"{MODEL_TYPE}-{FEATURE_VERSION}-{RUN_TYPE}-{timestamp}"

    # 구조화된 경로 생성
    paths = get_model_paths(run_name, timestamp)
    paths['experiment_dir'].mkdir(parents=True, exist_ok=True)

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
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
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

        # 구조화된 경로에 모델 저장
        model_path = paths['model']
        best_model.save_model(str(model_path))
        print(f"Model saved to: {model_path}")

        csv_path, plot_path = log_feature_importance(best_model, paths)

        results = {
            'cv_score_mean': float(mean_score),
            'cv_score_std': float(std_score),
            'cv_scores': [float(s) for s in cv_scores],
            'best_fold': int(best_fold_idx + 1),
            'xgb_params': {k: v for k, v in xgb_params.items() if k != 'device'}
        }
        results_path = paths['results']
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

        # 실험 요약 출력
        print(f"\n📁 Experiment saved in: {paths['experiment_dir']}")
        print(f"   - Model: {model_path.name}")
        print(f"   - Results: {results_path.name}")
        print(f"   - Feature importance: {csv_path.name}, {plot_path.name}")

        # WandB artifact에 전체 실험 폴더 추가
        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(str(model_path))
        artifact.add_file(str(csv_path))
        artifact.add_file(str(plot_path))
        artifact.add_file(str(results_path))
        run.log_artifact(artifact)

    finally:
        run.finish()
        print(f"\nTraining pipeline finished in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
