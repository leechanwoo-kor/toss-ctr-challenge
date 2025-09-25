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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.models.deepfm_core import DeepFMDataset
from src.utils.config import DEEPFM_CONFIG
from src.utils.sequence_utils import process_sequence_features

# Constants
DATA_DIR = "data/processed/deepfm"
XGBOOST_DATA_DIR = "data/processed/xgboost"
BATCH_SIZE = 500000


def load_xgboost_features():
    """XGBoost 처리된 피처 목록 로드"""
    try:
        with open(f"{XGBOOST_DATA_DIR}/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        with open(f"{XGBOOST_DATA_DIR}/processor.pkl", "rb") as f:
            processor = pickle.load(f)

        print(f"Loaded XGBoost features: {len(feature_names)} features")

        categorical = processor.categorical_features
        numerical = [f for f in feature_names if f not in categorical]

        print(f"  Categorical: {len(categorical)} features - {categorical}")
        print(f"  Numerical: {len(numerical)} features (includes EDA features)")

        return feature_names, categorical, numerical, processor

    except FileNotFoundError as e:
        raise FileNotFoundError(
            "XGBoost data not found. Run XGBoost data preparation first."
        ) from e




def create_deepfm_features(df):
    """XGBoost 피처 + 시퀀스 통계 피처 생성"""
    print("Creating XGBoost + sequence features...")

    # 1. XGBoost 피처 정보 로드
    xgb_features, xgb_categorical, xgb_numerical, xgb_processor = (
        load_xgboost_features()
    )

    # 2. 시퀀스 피처 먼저 처리 (utils 모듈 사용)
    seq_features = process_sequence_features(df["seq"], BATCH_SIZE)

    # 3. 원본 데이터에서 XGBoost Enhanced 피처만 선택
    print("🔍 Selecting exact XGBoost Enhanced features...")

    # XGBoost 데이터 직접 로드해서 변환 방식 복사
    xgb_X = np.load("data/processed/xgboost/X_train.npy")
    xgb_y = np.load("data/processed/xgboost/y_train.npy")

    print(f"   XGBoost shape: {xgb_X.shape}")
    print(f"   Target features: {len(xgb_features)}")

    # XGBoost 프로세서를 사용해서 동일하게 변환
    print("🔧 Applying XGBoost processor...")
    df_processed = xgb_processor.transform(df.copy())

    # 정확한 피처만 선택
    available_features = [f for f in xgb_features if f in df_processed.columns]
    missing_features = [f for f in xgb_features if f not in df_processed.columns]

    print(f"   Available features: {len(available_features)}/{len(xgb_features)}")
    if missing_features:
        print(f"   Missing features: {missing_features}")

    df_selected = df_processed[available_features + ["clicked"]].copy()

    # 4. 시퀀스 피처 추가
    print("🔗 Adding sequence features...")
    seq_features.index = df_selected.index
    df_final = pd.concat([df_selected, seq_features], axis=1)

    # 5. DeepFM용 스케일링 (수치형만)
    print("⚖️ Applying DeepFM scaling...")
    categorical_features = [f for f in available_features if f in xgb_categorical]
    numerical_features = [
        f for f in df_final.columns if f not in categorical_features and f != "clicked"
    ]

    # 스케일러 적용
    scaler = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=1000,
        subsample=100000,
        random_state=42,
    )

    df_final[numerical_features] = scaler.fit_transform(df_final[numerical_features])
    df_final[numerical_features] = df_final[numerical_features].astype(np.float32)

    print(f"✅ Exact DeepFM features created!")
    print(f"   Total features: {len(df_final.columns) - 1}")  # -1 for clicked
    print(f"   Categorical: {len(categorical_features)}")
    print(f"   Numerical: {len(numerical_features)}")
    print(f"   Expected: 78 + 7 = 85")

    return df_final, categorical_features, numerical_features, scaler, xgb_processor


class ExactDeepFMProcessor:
    """정확한 XGBoost Enhanced + 시퀀스 피처 프로세서"""

    def __init__(self, categorical_features, numerical_features, scaler, xgb_processor):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.scaler = scaler
        self.xgb_processor = xgb_processor

        # categorical_feature_dims 계산 (DeepFMDataset에서 필요)
        self.categorical_feature_dims = {}

    def transform(self, df):
        """테스트 데이터에 동일한 전처리 적용"""
        print("🔧 Applying ExactDeepFMProcessor transform...")

        # 1. 시퀀스 피처 처리
        seq_features = process_sequence_features(df["seq"])

        # 2. XGBoost 프로세서 적용
        print("   Applying XGBoost processor...")
        df_processed = self.xgb_processor.transform(df.copy())

        # 3. 정확한 피처만 선택
        available_features = [f for f in df_processed.columns
                             if f in self.categorical_features or f in self.numerical_features]

        # clicked 컬럼이 있다면 제외
        if "clicked" in df_processed.columns:
            df_selected = df_processed[available_features].copy()
        else:
            df_selected = df_processed[available_features].copy()

        # 4. 시퀀스 피처 추가
        print("   Adding sequence features...")
        seq_features.index = df_selected.index
        df_final = pd.concat([df_selected, seq_features], axis=1)

        # 5. 스케일링 (수치형만)
        print("   Applying scaling...")
        numerical_cols = [col for col in self.numerical_features if col in df_final.columns]
        if numerical_cols:
            df_final[numerical_cols] = self.scaler.transform(df_final[numerical_cols])
            df_final[numerical_cols] = df_final[numerical_cols].astype(np.float32)

        print(f"   Transformed data shape: {df_final.shape}")
        return df_final


def main():
    print("Starting DeepFM data preparation...")
    print("XGBoost features + sequence statistics features")

    start_time = time.time()

    try:
        # 1. 데이터 로드
        print("\nLoading raw data...")
        all_train = pd.read_parquet("data/train.parquet", engine="pyarrow")
        print(f"Training data loaded: {all_train.shape}")

        # 샘플링 (필요시)
        sample_size = DEEPFM_CONFIG.get("SAMPLE_SIZE")
        if sample_size:
            all_train = all_train.sample(
                n=sample_size, random_state=DEEPFM_CONFIG.get("SEED", 42)
            )
            print(f"Sampled to: {all_train.shape}")

        # 2. 피처 생성
        print("\nCreating features...")
        processed_df, categorical_features, numerical_features, scaler, xgb_processor = (
            create_deepfm_features(all_train)
        )

    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e

    # 메모리 정리
    del all_train
    gc.collect()

    # 3. 타겟과 피처 분리
    print("\nSeparating features and target...")
    X = processed_df.drop("clicked", axis=1)
    y = processed_df["clicked"].values

    print(f"Final feature shape: {X.shape}")
    print(f"CTR: {y.mean():.4f}")

    # 4. 훈련/검증 분할
    print("\n🔄 Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=DEEPFM_CONFIG["SEED"], stratify=y
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 5. 프로세서 생성 (xgb_processor 포함)
    processor = ExactDeepFMProcessor(categorical_features, numerical_features, scaler, xgb_processor)

    # categorical_feature_dims 계산
    for feat in categorical_features:
        if feat in X.columns:
            processor.categorical_feature_dims[feat] = X[feat].nunique() + 1

    # 6. DeepFM Dataset 생성 (간단한 방식)
    print("\n🧠 Creating DeepFM datasets...")

    # target과 함께 DataFrame 생성
    X_train_with_target = X_train.copy()
    X_train_with_target["clicked"] = y_train

    X_val_with_target = X_val.copy()
    X_val_with_target["clicked"] = y_val

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
        "categorical_features": categorical_features,
        "categorical_feature_dims": processor.categorical_feature_dims,
        "numerical_features": numerical_features,
        "data_shape": {
            "train": X_train.shape,
            "val": X_val.shape,
            "features": len(X_train.columns),
        },
        "ctr_stats": {
            "train_ctr": y_train.mean(),
            "val_ctr": y_val.mean(),
            "total_ctr": y.mean(),
        },
        "config_used": DEEPFM_CONFIG.copy(),
        "comparison_info": {
            "purpose": "XGBoost compatibility",
            "base_features": len(categorical_features)
            + len([f for f in numerical_features if not f.startswith("seq_")]),
            "sequence_features": len(
                [f for f in numerical_features if f.startswith("seq_")]
            ),
            "total_features": len(X_train.columns),
        },
    }

    # 8. 저장
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\nSaving to {DATA_DIR}...")

    # 데이터셋 저장
    torch.save(
        {
            "categorical": train_categorical,
            "numerical": train_numerical,
            "targets": train_targets,
        },
        f"{DATA_DIR}/train_dataset.pt",
    )

    torch.save(
        {
            "categorical": val_categorical,
            "numerical": val_numerical,
            "targets": val_targets,
        },
        f"{DATA_DIR}/val_dataset.pt",
    )

    # 메타데이터와 프로세서 저장
    with open(f"{DATA_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with open(f"{DATA_DIR}/processor.pkl", "wb") as f:
        pickle.dump(processor, f)

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("DeepFM data preparation completed!")
    print("=" * 50)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Features: {metadata['data_shape']['features']}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Numerical: {len(numerical_features)}")
    print(f"  Base (XGBoost): {metadata['comparison_info']['base_features']}")
    print(f"  Sequence: {metadata['comparison_info']['sequence_features']}")
    print(f"Train samples: {len(y_train):,}")
    print(f"Val samples: {len(y_val):,}")
    print(f"Saved to: {DATA_DIR}")

    return metadata


if __name__ == "__main__":
    try:
        metadata = main()
        print("\nSuccess! Ready for DeepFM training.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
