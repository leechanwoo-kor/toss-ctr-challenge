"""
XGBoost Optuna 하이퍼파라미터 최적화
"""

import os
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
import wandb

from ..utils.metrics import calculate_competition_metrics
from ..utils.config import XGBOOST_CONFIG, XGBOOST_WANDB_CONFIG


def load_prepared_data():
    """전처리된 데이터 로드"""
    print("📁 Loading prepared data...")

    data_dir = 'data/processed/xgboost'

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

    # 메타데이터 로드
    with open(f'{data_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # 피처명 로드
    with open(f'{data_dir}/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # 프로세서 로드
    with open(f'{data_dir}/processor.pkl', 'rb') as f:
        processor = pickle.load(f)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features: {len(feature_names)}")

    return X_train, X_val, y_train, y_val, feature_names, metadata, processor


def create_objective(X_train, X_val, y_train, y_val, feature_names, metadata):
    """Optuna objective 함수 생성"""

    def objective(trial):
        # 하이퍼파라미터 탐색 공간 정의
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': 'gbtree',
            'device': 'cuda:0',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42,
            'n_jobs': -1,

            # 탐색할 하이퍼파라미터
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 1000, 4000, step=500),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 50, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.3),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.3),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 40.0, 60.0),
        }

        # DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        # Pruning callback
        pruning_callback = XGBoostPruningCallback(trial, 'val-logloss')

        # 훈련 설정
        train_params = params.copy()
        n_estimators = train_params.pop('n_estimators')

        # 훈련 실행
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )

        # 예측 및 평가
        val_preds = model.predict(dval, iteration_range=(0, model.best_iteration))
        val_metrics = calculate_competition_metrics(y_val, val_preds)

        # Competition Score를 최대화 (Optuna는 최소화하므로 음수로 반환)
        return -val_metrics['comp_score']

    return objective


def main():
    print("🚀 Starting XGBoost Optuna optimization...")

    # 1. 데이터 로드
    start_time = time.time()
    X_train, X_val, y_train, y_val, feature_names, metadata, processor = load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # 2. WandB 초기화
    run_name = f"xgboost-optuna-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=f"{XGBOOST_WANDB_CONFIG['PROJECT']}-optuna",
        entity=XGBOOST_WANDB_CONFIG['ENTITY'],
        name=run_name,
        tags=XGBOOST_WANDB_CONFIG['TAGS'] + ['optuna', 'hyperparameter-optimization'],
        notes="XGBoost hyperparameter optimization with Optuna"
    )

    # 메타데이터 로깅
    wandb.log({
        'data_load_time': load_time,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'num_features': len(feature_names),
        'num_categorical': metadata['num_categorical'],
        'num_numerical': metadata['num_numerical'],
        'train_ctr': metadata['train_ctr'],
        'val_ctr': metadata['val_ctr']
    })

    # 3. Optuna Study 생성
    print("🔍 Creating Optuna study...")
    study = optuna.create_study(
        direction='maximize',  # Competition Score 최대화
        study_name=f"xgboost-{datetime.now().strftime('%m%d_%H%M')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    )

    # 4. Objective 함수 생성
    objective = create_objective(X_train, X_val, y_train, y_val, feature_names, metadata)

    # 5. 최적화 실행
    print("🎯 Starting hyperparameter optimization...")
    print("Configuration:")
    print(f"  - Trials: 50")
    print(f"  - Timeout: 4 hours")
    print(f"  - Pruning: MedianPruner")
    print(f"  - Objective: Competition Score (maximize)")

    optim_start = time.time()

    def log_callback(study, trial):
        """각 trial 결과를 WandB에 로깅"""
        wandb.log({
            f"trial_{trial.number}_comp_score": -trial.value,
            f"trial_{trial.number}_params": trial.params,
            "best_comp_score": -study.best_value,
            "completed_trials": len(study.trials)
        })

        print(f"Trial {trial.number}: Competition Score = {-trial.value:.6f}")
        if trial.number > 0:
            print(f"Best so far: {-study.best_value:.6f}")

    study.optimize(
        objective,
        n_trials=50,
        timeout=4 * 3600,  # 4시간
        callbacks=[log_callback]
    )

    optim_time = time.time() - optim_start

    # 6. 최적 결과 출력
    print("\n" + "="*60)
    print("🎉 XGBoost Optuna Optimization Completed!")
    print("="*60)
    print(f"Optimization Time: {optim_time:.2f}s ({optim_time/3600:.2f}h)")
    print(f"Completed Trials: {len(study.trials)}")
    print(f"Best Competition Score: {-study.best_value:.6f}")

    print(f"\n🏆 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 7. 최고 성능 모델로 재훈련 및 저장
    print("\n🔄 Retraining best model...")
    best_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'device': 'cuda:0',
        'tree_method': 'hist',
        'verbosity': 0,
        'random_state': 42,
        'n_jobs': -1,
        **study.best_params
    }

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # 최종 훈련
    train_params = best_params.copy()
    n_estimators = train_params.pop('n_estimators')

    evals = [(dtrain, 'train'), (dval, 'val')]
    best_model = xgb.train(
        train_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # 예측 및 평가
    train_preds = best_model.predict(dtrain, iteration_range=(0, best_model.best_iteration))
    val_preds = best_model.predict(dval, iteration_range=(0, best_model.best_iteration))

    train_metrics = calculate_competition_metrics(y_train, train_preds)
    val_metrics = calculate_competition_metrics(y_val, val_preds)

    # 8. 모델 저장
    print("💾 Saving optimized model...")
    model_dir = f"checkpoints/xgboost/{run_name}"
    os.makedirs(model_dir, exist_ok=True)

    # 모델 저장
    model_path = f"{model_dir}/xgboost_model.json"
    best_model.save_model(model_path)

    # 메타데이터 저장
    model_metadata = {
        'config': best_params,
        'feature_names': feature_names,
        'processor': processor,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_iteration': best_model.best_iteration,
        'optuna_study': {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_time': optim_time
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(f"{model_dir}/model_metadata.pkl", 'wb') as f:
        pickle.dump(model_metadata, f)

    # Optuna study 저장
    optuna_path = f"{model_dir}/optuna_study.pkl"
    with open(optuna_path, 'wb') as f:
        pickle.dump(study, f)

    # 9. 피처 중요도 분석
    importance_dict = best_model.get_score(importance_type='gain')
    feature_imp = pd.DataFrame([
        {'feature': feature, 'importance': importance_dict.get(feature, 0)}
        for feature in feature_names
    ]).sort_values('importance', ascending=False)

    print("Top 10 most important features:")
    print(feature_imp.head(10))
    feature_imp.to_csv(f"{model_dir}/feature_importance.csv", index=False)

    # 10. 최종 결과
    total_time = time.time() - start_time

    print(f"\n📊 Final Training Metrics:")
    print(f"  Competition Score: {train_metrics['comp_score']:.6f}")
    print(f"  AUC: {train_metrics['auc']:.6f}")
    print(f"  AP: {train_metrics['ap']:.6f}")
    print(f"  WLL: {train_metrics['wll']:.6f}")

    print(f"\n🎯 Final Validation Metrics:")
    print(f"  Competition Score: {val_metrics['comp_score']:.6f}")
    print(f"  AUC: {val_metrics['auc']:.6f}")
    print(f"  AP: {val_metrics['ap']:.6f}")
    print(f"  WLL: {val_metrics['wll']:.6f}")

    print(f"\nModel saved: {model_path}")
    print(f"Optuna study saved: {optuna_path}")

    # 성능 비교
    baseline_score = 0.345132  # 이전 최고 성능
    improvement = val_metrics['comp_score'] - baseline_score
    print(f"\n📈 Performance vs Baseline:")
    print(f"Baseline: {baseline_score:.6f}")
    print(f"Optimized: {val_metrics['comp_score']:.6f}")
    print(f"Improvement: {improvement:+.6f} ({improvement/baseline_score*100:+.2f}%)")

    if val_metrics['comp_score'] >= 0.35:
        print("🎯 TARGET ACHIEVED: 0.35+ Competition Score!")
    else:
        print(f"🎯 Target: 0.35+ (current: {val_metrics['comp_score']:.6f})")

    # 최종 메트릭 로깅
    wandb.log({
        'final_best_iteration': best_model.best_iteration,
        'final_train_comp_score': train_metrics['comp_score'],
        'final_val_comp_score': val_metrics['comp_score'],
        'final_train_auc': train_metrics['auc'],
        'final_val_auc': val_metrics['auc'],
        'optimization_time': optim_time,
        'total_time': total_time,
        'n_trials_completed': len(study.trials),
        'improvement_vs_baseline': improvement,
        'improvement_percentage': improvement/baseline_score*100
    })

    wandb.finish()

    return best_model, val_metrics, study


if __name__ == "__main__":
    main()