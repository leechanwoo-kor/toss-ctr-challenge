"""
XGBoost K-Fold Optuna 하이퍼파라미터 최적화
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import wandb

# Optuna XGBoost warning 숨기기
warnings.filterwarnings("ignore", message="The reported value is ignored")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.metrics import calculate_competition_metrics
from src.utils.config import XGBOOST_CONFIG, XGBOOST_WANDB_CONFIG, KFOLD_CONFIG


def load_prepared_data():
    """전처리된 데이터 로드"""
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

    return train_X, train_y, val_X, val_y, feature_names


def objective(trial):
    """Optuna 목적 함수 - K-fold CV로 하이퍼파라미터 최적화"""

    print(f"\n🔍 Trial {trial.number} starting...")

    # 하이퍼파라미터 제안 - "Less is More" 전략
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "device": "cuda:0",
        "tree_method": "hist",

        # 🎯 현재 최고 성능(120 iter, max_depth=8) 주변 정밀 탐색
        "max_depth": trial.suggest_int("max_depth", 6, 10),                    # 6-10으로 확장
        "learning_rate": trial.suggest_float("learning_rate", 0.08, 0.15),     # 0.1 주변
        "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),   # 80-200으로 현실적
        "subsample": trial.suggest_float("subsample", 0.85, 0.95),             # 0.9 주변
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.85, 0.95), # 0.9 주변
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.85, 0.95), # 0.9 주변
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),        # 1-5로 현실적
        "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 0.2),              # 0.1 주변
        "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 0.2),            # 0.1 주변

        # 고정 파라미터들
        "scale_pos_weight": 51.6,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # K-Fold 설정
    kf = StratifiedKFold(
        n_splits=KFOLD_CONFIG["n_splits"],
        shuffle=KFOLD_CONFIG["shuffle"],
        random_state=KFOLD_CONFIG["random_state"]
    )

    cv_scores = []

    # K-fold CV 실행
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"    Training fold {fold + 1}/{KFOLD_CONFIG['n_splits']}...")

        # Split data for this fold
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_valid_fold = X_train.iloc[valid_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        # Pruning callback 복구 (warning은 숨김 처리됨)
        pruning_callback = XGBoostPruningCallback(trial, f"valid-logloss")

        # Train model
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=params["n_estimators"],
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=30,  # 빠른 수렴 유도
                verbose_eval=False,
                callbacks=[pruning_callback]
            )

            # Predict and evaluate
            dvalid_pred = xgb.DMatrix(X_valid_fold)
            fold_pred = model.predict(dvalid_pred)

            # Competition score 계산
            fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
            cv_scores.append(fold_metrics['comp_score'])

        except optuna.TrialPruned:
            # Pruned trial - 조기 종료
            raise optuna.TrialPruned()
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            # 실패한 fold는 평균값으로 대체
            cv_scores.append(0.32)  # 보수적 점수

    # K-fold CV 평균 점수
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # 🎯 더 균형잡힌 목적 함수
    # CV 표준편차 페널티를 줄임
    generalization_penalty = cv_std * 1.0  # 표준편차 1배 페널티 (기존 3배에서 감소)
    objective_score = cv_mean - generalization_penalty

    # WandB에 trial 결과 로깅
    wandb.log({
        f"trial_{trial.number}_cv_score": cv_mean,
        f"trial_{trial.number}_cv_std": cv_std,
        f"trial_{trial.number}_objective_score": objective_score,
        f"trial_{trial.number}_penalty": generalization_penalty,
        **{f"trial_{trial.number}_{key}": value for key, value in trial.params.items()}
    })

    print(f"Trial {trial.number}: CV Score = {cv_mean:.6f} ± {cv_std:.6f}, Objective = {objective_score:.6f}")

    return objective_score  # 페널티 적용된 점수 반환


def main():
    """메인 함수"""
    global X_train, y_train, X_val, y_val

    print("🚀 Starting XGBoost K-Fold Optuna Optimization...")

    # 데이터 로드
    start_time = time.time()
    X_train, y_train, X_val, y_val, feature_names = load_prepared_data()
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # WandB 초기화
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"xgboost-optuna-kfold-{timestamp}"

    wandb.init(
        project=f"{XGBOOST_WANDB_CONFIG['PROJECT']}-optuna",
        entity=XGBOOST_WANDB_CONFIG['ENTITY'],
        name=run_name,
        tags=XGBOOST_WANDB_CONFIG['TAGS'] + ["optuna", "kfold", "hyperopt"],
        notes="XGBoost K-Fold hyperparameter optimization with Optuna",
        config={
            "num_features": len(feature_names),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_ctr": y_train.mean(),
            "val_ctr": y_val.mean(),
            "kfold_splits": KFOLD_CONFIG["n_splits"],
        }
    )

    # Optuna study 생성
    study_name = f"xgboost-kfold-{timestamp}"
    storage = f"sqlite:///optuna_studies/{study_name}.db"
    os.makedirs("optuna_studies", exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Competition score 최대화
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=20
        ),
        load_if_exists=True
    )

    print(f"📊 Optuna Study: {study_name}")
    print(f"🎯 Target: Maximize Competition Score")
    print(f"🔄 K-Fold: {KFOLD_CONFIG['n_splits']}-fold Cross Validation")

    # 최적화 실행 - 현재 최고 성능 주변 정밀 탐색
    n_trials = 50  # 정밀 탐색으로 최적해 발견
    print(f"🚀 Starting optimization with {n_trials} trials...")
    print(f"🎯 Strategy: Fine-tuning around current best performance (120 iter, max_depth=8)")

    try:
        study.optimize(objective, n_trials=n_trials, timeout=3600*4)  # 4시간 제한
    except KeyboardInterrupt:
        print("⚠️ Optimization interrupted by user")

    # 최적 결과
    best_trial = study.best_trial
    print(f"\n🎉 Best trial: {best_trial.number}")
    print(f"🏆 Best CV Score: {best_trial.value:.6f}")
    print(f"📋 Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # 최적 파라미터로 최종 학습 및 평가
    print(f"\n🔄 Training final model with best parameters...")

    best_params = best_trial.params.copy()
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "device": "cuda:0",
        "tree_method": "hist",
        "scale_pos_weight": 51.6,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    })

    # 최종 K-fold 학습
    kf = StratifiedKFold(
        n_splits=KFOLD_CONFIG["n_splits"],
        shuffle=KFOLD_CONFIG["shuffle"],
        random_state=KFOLD_CONFIG["random_state"]
    )

    final_models = []
    final_cv_scores = []
    val_predictions = np.zeros(len(y_val))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Final training fold {fold + 1}/{KFOLD_CONFIG['n_splits']}...")

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_valid_fold = X_train.iloc[valid_idx]
        y_valid_fold = y_train.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)

        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_params["n_estimators"],
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=30,  # 일관된 early stopping
            verbose_eval=False
        )

        final_models.append(model)

        # Validation set 예측
        dval = xgb.DMatrix(X_val)
        val_predictions += model.predict(dval) / KFOLD_CONFIG["n_splits"]

        # Fold 성능
        fold_pred = model.predict(dvalid)
        fold_metrics = calculate_competition_metrics(y_valid_fold, fold_pred)
        final_cv_scores.append(fold_metrics['comp_score'])

    # 최종 평가
    final_metrics = calculate_competition_metrics(y_val, val_predictions)

    print(f"\n🎯 Final Results:")
    print(f"CV Score: {np.mean(final_cv_scores):.6f} ± {np.std(final_cv_scores):.6f}")
    print(f"Validation Score: {final_metrics['comp_score']:.6f}")
    print(f"Validation AUC: {final_metrics['auc']:.6f}")

    # 결과 저장
    result_dir = f"checkpoints/xgboost/{run_name}"
    os.makedirs(result_dir, exist_ok=True)

    for i, model in enumerate(final_models):
        model.save_model(f"{result_dir}/xgboost_fold_{i+1}.json")

    # 최적 파라미터 저장
    with open(f"{result_dir}/best_params.pkl", 'wb') as f:
        pickle.dump(best_params, f)

    # Study 정보 저장
    study_df = study.trials_dataframe()
    study_df.to_csv(f"{result_dir}/optuna_trials.csv", index=False)

    # WandB 최종 로깅
    wandb.log({
        "best_cv_score": best_trial.value,
        "final_cv_mean": np.mean(final_cv_scores),
        "final_cv_std": np.std(final_cv_scores),
        "final_val_score": final_metrics['comp_score'],
        "final_val_auc": final_metrics['auc'],
        "n_trials_completed": len(study.trials),
        **{f"best_{key}": value for key, value in best_trial.params.items()}
    })

    wandb.finish()

    target_achieved = final_metrics['comp_score'] >= 0.35
    if target_achieved:
        print(f"🎉 TARGET ACHIEVED: {final_metrics['comp_score']:.6f} >= 0.35")
    else:
        print(f"🎯 Target: 0.35+ (current: {final_metrics['comp_score']:.6f})")

    print(f"\n💾 Results saved: {result_dir}")


if __name__ == "__main__":
    main()