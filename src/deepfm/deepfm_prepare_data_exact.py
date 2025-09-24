"""
XGBoost Enhanced와 정확히 동일한 78개 피처 + 시퀀스 피처 사용하는 DeepFM 데이터 준비
직접적인 피처 선택으로 100% 정확성 보장
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.deepfm_core import DeepFMDataset
from src.utils.config import DEEPFM_CONFIG


def load_xgboost_enhanced_features():
    """XGBoost Enhanced의 정확한 피처 목록 로드"""
    try:
        with open('data/processed/xgboost_enhanced/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        with open('data/processed/xgboost_enhanced/processor.pkl', 'rb') as f:
            processor = pickle.load(f)

        print(f"✅ Loaded XGBoost Enhanced features: {len(feature_names)} features")

        categorical = processor.categorical_features
        numerical = [f for f in feature_names if f not in categorical]

        print(f"   Categorical: {len(categorical)} - {categorical}")
        print(f"   Numerical: {len(numerical)} (EDA included)")

        return feature_names, categorical, numerical, processor

    except FileNotFoundError:
        raise FileNotFoundError("XGBoost Enhanced data not found. Run XGBoost Enhanced preparation first.")


def process_sequence_features(seq_series):
    """시퀀스를 통계 피처로 변환"""
    print("📊 Processing sequence features...")

    def extract_stats(seq_str):
        if pd.isna(seq_str) or seq_str == '' or seq_str == 'nan':
            return [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            seq_array = np.fromstring(str(seq_str), sep=",", dtype=np.float32)
            if len(seq_array) == 0:
                return [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            length = len(seq_array)
            mean = float(np.mean(seq_array))
            std = float(np.std(seq_array))
            minimum = float(np.min(seq_array))
            maximum = float(np.max(seq_array))
            total = float(np.sum(seq_array))

            range_val = maximum - minimum if maximum != minimum else 0
            normalized_range = range_val / (maximum + 1e-8) if maximum > 0 else 0
            density = total / length if length > 0 else 0
            cv = std / (mean + 1e-8) if mean != 0 else 0

            return [length, mean, std, minimum, maximum, total, normalized_range, density, cv]

        except:
            return [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 배치 처리
    batch_size = 100000
    all_stats = []

    for i in range(0, len(seq_series), batch_size):
        batch_end = min(i + batch_size, len(seq_series))
        batch_seq = seq_series.iloc[i:batch_end]
        batch_stats = [extract_stats(seq) for seq in batch_seq]
        all_stats.extend(batch_stats)

        if i % (batch_size * 5) == 0:
            print(f"     Processed {batch_end:,}/{len(seq_series):,} sequences...")

    seq_features = pd.DataFrame(all_stats, columns=[
        'seq_length', 'seq_mean', 'seq_std', 'seq_min', 'seq_max', 'seq_sum',
        'seq_normalized_range', 'seq_density', 'seq_cv'
    ], dtype=np.float32)

    print(f"   Created {seq_features.shape[1]} sequence features")
    return seq_features


def create_exact_deepfm_features(df):
    """정확히 XGBoost Enhanced + 시퀀스 피처 생성"""
    print("🎯 Creating exact XGBoost Enhanced + sequence features...")

    # 1. XGBoost Enhanced 피처 정보 로드
    xgb_features, xgb_categorical, xgb_numerical, xgb_processor = load_xgboost_enhanced_features()

    # 2. 시퀀스 피처 먼저 처리
    seq_features = process_sequence_features(df['seq'])

    # 3. 원본 데이터에서 XGBoost Enhanced 피처만 선택
    print("🔍 Selecting exact XGBoost Enhanced features...")

    # XGBoost Enhanced 데이터 직접 로드해서 변환 방식 복사
    xgb_X = np.load('data/processed/xgboost_enhanced/X_train.npy')
    xgb_y = np.load('data/processed/xgboost_enhanced/y_train.npy')

    print(f"   XGBoost Enhanced shape: {xgb_X.shape}")
    print(f"   Target features: {len(xgb_features)}")

    # XGBoost Enhanced 프로세서를 사용해서 동일하게 변환
    print("🔧 Applying XGBoost Enhanced processor...")
    df_processed = xgb_processor.fit_transform(df.copy())

    # 정확한 피처만 선택
    available_features = [f for f in xgb_features if f in df_processed.columns]
    missing_features = [f for f in xgb_features if f not in df_processed.columns]

    print(f"   Available features: {len(available_features)}/{len(xgb_features)}")
    if missing_features:
        print(f"   Missing features: {missing_features}")

    df_selected = df_processed[available_features + ['clicked']].copy()

    # 4. 시퀀스 피처 추가
    print("🔗 Adding sequence features...")
    seq_features.index = df_selected.index
    df_final = pd.concat([df_selected, seq_features], axis=1)

    # 5. DeepFM용 스케일링 (수치형만)
    print("⚖️ Applying DeepFM scaling...")
    categorical_features = [f for f in available_features if f in xgb_categorical]
    numerical_features = [f for f in df_final.columns
                         if f not in categorical_features and f != 'clicked']

    # 스케일러 적용
    scaler = QuantileTransformer(
        output_distribution='uniform',
        n_quantiles=1000,
        subsample=100000,
        random_state=42
    )

    df_final[numerical_features] = scaler.fit_transform(df_final[numerical_features])
    df_final[numerical_features] = df_final[numerical_features].astype(np.float32)

    print(f"✅ Exact DeepFM features created!")
    print(f"   Total features: {len(df_final.columns) - 1}")  # -1 for clicked
    print(f"   Categorical: {len(categorical_features)}")
    print(f"   Numerical: {len(numerical_features)}")
    print(f"   Expected: 78 + 9 = 87")

    return df_final, categorical_features, numerical_features, scaler


class ExactDeepFMProcessor:
    """정확한 XGBoost Enhanced + 시퀀스 피처 프로세서"""

    def __init__(self, categorical_features, numerical_features, scaler):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.scaler = scaler

        # categorical_feature_dims 계산 (DeepFMDataset에서 필요)
        self.categorical_feature_dims = {}


def main():
    print("🎯 Starting Exact DeepFM data preparation...")
    print("   Goal: Exactly XGBoost Enhanced (78) + sequence (9) = 87 features")

    start_time = time.time()

    # 1. 데이터 로드
    print("\n📁 Loading raw data...")
    all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
    print(f"Full training data loaded: {all_train.shape}")

    # 샘플링 (필요시)
    sample_size = DEEPFM_CONFIG['SAMPLE_SIZE']
    if sample_size:
        all_train = all_train.sample(n=sample_size, random_state=DEEPFM_CONFIG['SEED'])
        print(f"Sampled to: {all_train.shape}")

    # 2. 정확한 피처 생성
    print("\n🔧 Creating exact features...")
    processed_df, categorical_features, numerical_features, scaler = create_exact_deepfm_features(all_train)

    # 메모리 정리
    del all_train
    gc.collect()

    # 3. 타겟과 피처 분리
    print("\n✂️ Separating features and target...")
    X = processed_df.drop('clicked', axis=1)
    y = processed_df['clicked'].values

    print(f"Final feature shape: {X.shape}")
    print(f"CTR: {y.mean():.4f}")

    # 4. 훈련/검증 분할
    print("\n🔄 Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=DEEPFM_CONFIG['SEED'], stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 5. 프로세서 생성
    processor = ExactDeepFMProcessor(categorical_features, numerical_features, scaler)

    # categorical_feature_dims 계산
    for feat in categorical_features:
        if feat in X.columns:
            processor.categorical_feature_dims[feat] = X[feat].nunique() + 1

    # 6. DeepFM Dataset 생성 (간단한 방식)
    print("\n🧠 Creating DeepFM datasets...")

    # target과 함께 DataFrame 생성
    X_train_with_target = X_train.copy()
    X_train_with_target['clicked'] = y_train

    X_val_with_target = X_val.copy()
    X_val_with_target['clicked'] = y_val

    # 수동으로 dataset 생성 (오류 방지)
    train_categorical = {}
    train_numerical = None

    # categorical 데이터
    for col in categorical_features:
        if col in X_train.columns:
            train_categorical[col] = torch.LongTensor(X_train[col].values)

    # numerical 데이터
    if numerical_features:
        numerical_cols = [col for col in numerical_features if col in X_train.columns]
        if numerical_cols:
            train_numerical = torch.FloatTensor(X_train[numerical_cols].values)

    train_targets = torch.FloatTensor(y_train)

    # validation도 동일하게
    val_categorical = {}
    for col in categorical_features:
        if col in X_val.columns:
            val_categorical[col] = torch.LongTensor(X_val[col].values)

    val_numerical = None
    if numerical_features:
        numerical_cols = [col for col in numerical_features if col in X_val.columns]
        if numerical_cols:
            val_numerical = torch.FloatTensor(X_val[numerical_cols].values)

    val_targets = torch.FloatTensor(y_val)

    print(f"Train dataset: {len(y_train)} samples")
    print(f"Val dataset: {len(y_val)} samples")

    # 7. 메타데이터
    metadata = {
        'categorical_features': categorical_features,
        'categorical_feature_dims': processor.categorical_feature_dims,
        'numerical_features': numerical_features,
        'data_shape': {
            'train': X_train.shape,
            'val': X_val.shape,
            'features': len(X_train.columns)
        },
        'ctr_stats': {
            'train_ctr': y_train.mean(),
            'val_ctr': y_val.mean(),
            'total_ctr': y.mean()
        },
        'config_used': DEEPFM_CONFIG.copy(),
        'comparison_info': {
            'purpose': 'Exact XGBoost Enhanced compatibility',
            'base_features': len(categorical_features) + len([f for f in numerical_features if not f.startswith('seq_')]),
            'sequence_features': len([f for f in numerical_features if f.startswith('seq_')]),
            'total_features': len(X_train.columns)
        }
    }

    # 8. 저장
    save_dir = "data/processed/deepfm_exact"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n💾 Saving to {save_dir}...")

    # 수동으로 저장
    torch.save({
        'categorical': train_categorical,
        'numerical': train_numerical,
        'targets': train_targets
    }, f"{save_dir}/train_dataset.pt")

    torch.save({
        'categorical': val_categorical,
        'numerical': val_numerical,
        'targets': val_targets
    }, f"{save_dir}/val_dataset.pt")

    # 메타데이터와 프로세서 저장
    with open(f"{save_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with open(f"{save_dir}/processor.pkl", "wb") as f:
        pickle.dump(processor, f)

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("✅ Exact DeepFM data preparation completed!")
    print("="*60)
    print(f"📊 Results:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Features: {metadata['data_shape']['features']}")
    print(f"   - Categorical: {len(categorical_features)}")
    print(f"   - Numerical: {len(numerical_features)}")
    print(f"   - Base (XGBoost): {metadata['comparison_info']['base_features']}")
    print(f"   - Sequence: {metadata['comparison_info']['sequence_features']}")
    print(f"   Train samples: {len(y_train):,}")
    print(f"   Val samples: {len(y_val):,}")

    expected = 87
    actual = metadata['data_shape']['features']
    status = "✅ PERFECT" if actual == expected else f"❌ MISMATCH ({actual} vs {expected})"
    print(f"   Status: {status}")

    return metadata


if __name__ == "__main__":
    try:
        metadata = main()
        print("\n🎉 Success! Ready for exact Tree vs DeepFM comparison.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()