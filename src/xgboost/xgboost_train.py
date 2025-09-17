"""
XGBoost 훈련 스크립트 (현업 표준 설정)
"""

import os
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb
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


def main():
    print("🚀 Starting XGBoost training...")

    # 1. 전처리된 데이터 로드
    start_time = time.time()
    X_train, X_val, y_train, y_val, feature_names, metadata, processor = load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # 2. WandB 초기화
    run_name = f"xgboost-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=XGBOOST_WANDB_CONFIG['PROJECT'],
        entity=XGBOOST_WANDB_CONFIG['ENTITY'],
        name=run_name,
        config={**XGBOOST_CONFIG, **metadata['config_used']},
        tags=XGBOOST_WANDB_CONFIG['TAGS'],
        notes=XGBOOST_WANDB_CONFIG['NOTES']
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

    # 3. XGBoost DMatrix 생성 (GPU 최적화)
    print("🔄 Creating XGBoost DMatrix with GPU support...")
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=feature_names
    )

    dval = xgb.DMatrix(
        X_val,
        label=y_val,
        feature_names=feature_names
    )

    # GPU 사용 명시 (config에 gpu 설정이 있을 때만)
    if ('device' in XGBOOST_CONFIG and 'cuda' in XGBOOST_CONFIG['device']) or \
       ('tree_method' in XGBOOST_CONFIG and 'gpu' in XGBOOST_CONFIG['tree_method']):
        print("🚀 GPU acceleration enabled for training")
    else:
        print("💻 Using CPU for training")

    # 4. 모델 훈련
    print("🚀 Training XGBoost model...")
    print(f"📊 Training config:")
    for key, value in XGBOOST_CONFIG.items():
        if key != 'early_stopping_rounds':  # train에서 별도 처리
            print(f"  {key}: {value}")

    training_start = time.time()

    # early_stopping_rounds 분리
    train_config = XGBOOST_CONFIG.copy()
    early_stopping_rounds = train_config.pop('early_stopping_rounds', 100)

    # WandB 콜백 함수 생성
    def wandb_callback(env):
        """XGBoost evaluation 결과를 WandB에 실시간 로깅"""
        if env.iteration % 10 == 0:  # 10 iteration마다 로깅
            log_dict = {'iteration': env.iteration}

            # evaluation 결과 로깅
            for eval_name, eval_result in env.evaluation_result_list:
                metric_name = eval_result.split(':')[0] if ':' in eval_result else 'metric'
                metric_value = float(eval_result.split(':')[1]) if ':' in eval_result else float(eval_result)
                log_dict[f'{eval_name}_{metric_name}'] = metric_value

            wandb.log(log_dict)

    # 훈련 실행
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        train_config,
        dtrain,
        num_boost_round=train_config['n_estimators'],
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
        callbacks=[wandb_callback]  # WandB 콜백 추가
    )

    training_time = time.time() - training_start
    print(f"Training completed in {training_time:.2f} seconds")

    # 5. 예측 및 평가
    print("🔮 Generating predictions...")
    train_preds = model.predict(dtrain, iteration_range=(0, model.best_iteration))
    val_preds = model.predict(dval, iteration_range=(0, model.best_iteration))

    # 메트릭 계산
    train_metrics = calculate_competition_metrics(y_train, train_preds)
    val_metrics = calculate_competition_metrics(y_val, val_preds)

    # 6. 모델 저장
    print("💾 Saving model...")
    model_dir = f"checkpoints/xgboost/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/xgboost_model.json"

    model.save_model(model_path)

    # 메타데이터도 함께 저장
    model_metadata = {
        'config': XGBOOST_CONFIG,
        'feature_names': feature_names,
        'processor': processor,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_iteration': model.best_iteration,
        'timestamp': datetime.now().isoformat()
    }

    with open(f"{model_dir}/model_metadata.pkl", 'wb') as f:
        pickle.dump(model_metadata, f)

    # 7. 피처 중요도 분석
    print("📊 Analyzing feature importance...")
    importance_dict = model.get_score(importance_type='gain')

    feature_imp = pd.DataFrame([
        {'feature': feature, 'importance': importance_dict.get(feature, 0)}
        for feature in feature_names
    ]).sort_values('importance', ascending=False)

    print("Top 10 most important features:")
    print(feature_imp.head(10))

    # 피처 중요도 저장
    feature_imp.to_csv(f"{model_dir}/feature_importance.csv", index=False)

    # WandB에 Feature Importance 로깅
    wandb.log({
        "feature_importance": wandb.Table(
            dataframe=feature_imp.head(20),  # 상위 20개 피처
            columns=["feature", "importance"]
        )
    })

    # 예측 분포 히스토그램 로깅
    wandb.log({
        "train_predictions_hist": wandb.Histogram(train_preds),
        "val_predictions_hist": wandb.Histogram(val_preds),
        "train_predictions_mean": train_preds.mean(),
        "val_predictions_mean": val_preds.mean(),
        "train_predictions_std": train_preds.std(),
        "val_predictions_std": val_preds.std(),
    })

    # 8. 최종 결과
    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("🎉 XGBoost Training Completed!")
    print("="*50)
    print(f"Data Load Time: {load_time:.2f}s")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Best Iteration: {model.best_iteration}")

    print(f"\n📊 Training Metrics:")
    print(f"  Competition Score: {train_metrics['comp_score']:.6f}")
    print(f"  AUC: {train_metrics['auc']:.6f}")
    print(f"  AP: {train_metrics['ap']:.6f}")
    print(f"  WLL: {train_metrics['wll']:.6f}")

    print(f"\n🎯 Validation Metrics:")
    print(f"  Competition Score: {val_metrics['comp_score']:.6f}")
    print(f"  AUC: {val_metrics['auc']:.6f}")
    print(f"  AP: {val_metrics['ap']:.6f}")
    print(f"  WLL: {val_metrics['wll']:.6f}")

    print(f"\nModel saved: {model_path}")

    # 성능 비교 정보
    print(f"\n📈 Performance Summary:")
    print(f"Training time: {training_time:.2f}s (vs DeepFM ~2000s)")
    print(f"Competition Score: {val_metrics['comp_score']:.6f}")
    if val_metrics['comp_score'] >= 0.35:
        print("🎯 TARGET ACHIEVED: 0.35+ Competition Score!")
    else:
        print(f"🎯 Target: 0.35+ (current: {val_metrics['comp_score']:.6f})")

    # 최종 메트릭 로깅
    wandb.log({
        'best_iteration': model.best_iteration,
        'train_comp_score': train_metrics['comp_score'],
        'val_comp_score': val_metrics['comp_score'],
        'train_auc': train_metrics['auc'],
        'val_auc': val_metrics['auc'],
        'training_time': training_time,
        'total_time': total_time
    })

    wandb.finish()

    return model, val_metrics


if __name__ == "__main__":
    main()