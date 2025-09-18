"""
EDA 기반 새 피처들이 포함된 XGBoost용 데이터 준비 스크립트
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.enhanced_tree_processor import EnhancedTreeFeatureProcessor


def prepare_enhanced_xgboost_data():
    """EDA 기반 피처 엔지니어링이 적용된 XGBoost 데이터 준비"""
    print("🚀 Preparing enhanced XGBoost data with EDA features...")
    start_time = time.time()

    # 1. 원본 데이터 로드
    print("📁 Loading raw data...")
    all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
    print(f"Full training data loaded: {all_train.shape}")

    # 2. Enhanced 프로세서로 피처 처리
    print("🚀 Processing features with EDA enhancements...")
    processor = EnhancedTreeFeatureProcessor()

    # 타겟 분리 전에 EDA 피처 생성 (clicked 컬럼 필요)
    all_train_processed = processor.fit_transform(all_train.copy())
    print(f"Features processed: {all_train_processed.shape}")

    # 타겟 분리
    X_processed = all_train_processed.drop('clicked', axis=1)
    y = all_train_processed['clicked'].values

    # 3. 데이터 분할
    print("✂️ Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Train CTR: {y_train.mean():.4f}, Val CTR: {y_val.mean():.4f}")

    # 4. 메모리 최적화
    print("💾 Optimizing memory usage...")
    for col in X_train.columns:
        if X_train[col].dtype == 'int64':
            X_train[col] = X_train[col].astype('int32')
            X_val[col] = X_val[col].astype('int32')
        elif X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')

    # 5. 새 피처들 요약
    feature_groups = processor.get_feature_importance_groups()
    print(f"\n📊 Feature Summary:")
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in X_train.columns]
        print(f"  {group_name}: {len(available_features)} features")
        if group_name.startswith('eda') and available_features:
            print(f"    {available_features}")

    # 6. 저장
    output_dir = 'data/processed/xgboost_enhanced'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving to {output_dir}...")

    # 데이터 저장
    np.save(f'{output_dir}/X_train.npy', X_train.values)
    np.save(f'{output_dir}/X_val.npy', X_val.values)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_val.npy', y_val)

    # 피처명 저장
    feature_names = list(X_train.columns)
    with open(f'{output_dir}/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # 프로세서 저장
    with open(f'{output_dir}/processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    # 메타데이터 저장
    metadata = {
        'train_shape': X_train.shape,
        'val_shape': X_val.shape,
        'num_categorical': len(processor.categorical_features),
        'num_numerical': len(processor.numerical_features),
        'train_ctr': y_train.mean(),
        'val_ctr': y_val.mean(),
        'created_at': datetime.now().isoformat(),
        'processing_time': time.time() - start_time,
        'feature_groups': {k: len([f for f in v if f in feature_names])
                          for k, v in feature_groups.items()},
        'new_features': [f for f in processor.new_features if f in feature_names]
    }

    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # 메모리 정리
    import gc
    del X_processed, all_train
    gc.collect()

    # 새 피처들 통계
    print(f"\n📈 New Feature Statistics:")
    for feat in processor.new_features:
        if feat in X_train.columns:
            train_stats = X_train[feat].describe()
            print(f"  {feat}:")
            print(f"    mean={train_stats['mean']:.6f}, std={train_stats['std']:.6f}")
            print(f"    min={train_stats['min']:.6f}, max={train_stats['max']:.6f}")

    total_time = time.time() - start_time
    print(f"\n🎉 Enhanced data preparation completed!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📁 Saved to: {output_dir}")
    print(f"🆕 Added {len(processor.new_features)} new EDA features")

    # 임팩트 예상
    print(f"\n🎯 Expected Impact:")
    print(f"  Current baseline: ~0.353 CV")
    print(f"  With user segment CTR: +0.003~0.005")
    print(f"  With inventory CTR: +0.002~0.003")
    print(f"  With temporal features: +0.001~0.002")
    print(f"  Expected total: 0.359~0.363 CV (+0.006~0.010)")

    return metadata


if __name__ == "__main__":
    try:
        metadata = prepare_enhanced_xgboost_data()
        print("\n✅ Success! Ready for enhanced XGBoost training.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()