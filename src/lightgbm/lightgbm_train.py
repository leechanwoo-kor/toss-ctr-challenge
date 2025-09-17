"""
LightGBM 훈련 스크립트 (현업 표준 설정)
"""

import os
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import lightgbm as lgb
import wandb

from ..utils.metrics import calculate_competition_metrics
from ..utils.config import LIGHTGBM_CONFIG, LIGHTGBM_WANDB_CONFIG


def load_prepared_data():
    """전처리된 데이터 로드"""
    print("📁 Loading prepared data...")

    data_dir = 'data/processed/lightgbm'

    # 데이터 파일 존재 확인
    required_files = ['X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy',
                      'metadata.pkl', 'processor.pkl', 'feature_names.pkl']
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            raise FileNotFoundError(f"Required file not found: {file}. Run 'python -m src.lightgbm.lightgbm_prepare_data' first.")

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
    print("🌳 Starting LightGBM training...")

    # 1. 전처리된 데이터 로드
    start_time = time.time()
    X_train, X_val, y_train, y_val, feature_names, metadata, processor = load_prepared_data()
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")

    # 2. WandB 초기화
    run_name = f"lightgbm-{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=LIGHTGBM_WANDB_CONFIG['PROJECT'],
        entity=LIGHTGBM_WANDB_CONFIG['ENTITY'],
        name=run_name,
        config={**LIGHTGBM_CONFIG, **metadata['config_used']},
        tags=LIGHTGBM_WANDB_CONFIG['TAGS'],
        notes=LIGHTGBM_WANDB_CONFIG['NOTES']
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

    # 3. LightGBM 데이터셋 생성
    print("🔄 Creating LightGBM datasets...")
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_names,
        free_raw_data=False
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        feature_name=feature_names,
        reference=train_data,
        free_raw_data=False
    )

    # 4. 모델 훈련
    print("🌳 Training LightGBM model...")
    print(f"📊 Training config:")
    for key, value in LIGHTGBM_CONFIG.items():
        print(f"  {key}: {value}")

    training_start = time.time()

    # 콜백 함수 정의
    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=LIGHTGBM_CONFIG['early_stopping_rounds'])
    ]

    # 훈련 실행
    model = lgb.train(
        LIGHTGBM_CONFIG,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    training_time = time.time() - training_start
    print(f"Training completed in {training_time:.2f} seconds")

    # 5. 예측 및 평가
    print("🔮 Generating predictions...")
    train_preds = model.predict(X_train, num_iteration=model.best_iteration)
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    # 메트릭 계산
    train_metrics = calculate_competition_metrics(y_train, train_preds)
    val_metrics = calculate_competition_metrics(y_val, val_preds)

    # 6. 모델 저장
    print("💾 Saving model...")
    model_dir = f"checkpoints/lightgbm/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/lightgbm_model.txt"

    model.save_model(model_path)

    # 메타데이터도 함께 저장
    model_metadata = {
        'config': LIGHTGBM_CONFIG,
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
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print("Top 10 most important features:")
    print(feature_imp.head(10))

    # 피처 중요도 저장
    feature_imp.to_csv(f"{model_dir}/feature_importance.csv", index=False)

    # 8. 최종 결과
    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("🎉 LightGBM Training Completed!")
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