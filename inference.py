"""
Inference Script for Toss CTR Prediction Challenge
Reproduces October 12, 2025 Final Submission

This script:
1. Loads XGBoost K-Fold models from Oct 12 checkpoint
2. Loads DeepFM K-Fold models from Oct 10 checkpoint (used in Oct 12 submission)
3. Generates predictions from both models
4. Creates ensemble predictions with multiple weight combinations
5. Saves all submission files

Usage:
    python inference.py

Output:
    - submissions/xgboost/xgboost_reproduction.csv
    - submissions/deepfm/deepfm_reproduction.csv
    - submissions/ensemble/ensemble_reproduction_*.csv (6 weight combinations)
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.deepfm_dynamic_embedding import DynamicEmbeddingDeepFM
from src.deepfm.deepfm_processor import DeepFMProcessor

# ==============================================================================
# IMPORTANT: Checkpoint Paths for Oct 12, 2025 Submission Reproduction
# ==============================================================================
DEEPFM_CHECKPOINT = "checkpoints/deepfm_kfold/deepfm-dynamic-embedding-kfold-1010_1642"
XGBOOST_CHECKPOINT = "checkpoints/xgboost/xgboost-kfold-1012_1718"  # Oct 12 checkpoint

# Data paths
TEST_DATA_PATH = "data/test.parquet"
DEEPFM_PROCESSOR_PATH = "data/processed/deepfm_new/processor.pkl"
XGBOOST_PROCESSOR_PATH = "data/processed/xgboost/processor.pkl"
XGBOOST_FEATURES_PATH = "data/processed/xgboost/feature_names.pkl"

# Output paths
OUTPUT_DIR = "submissions/reproduction"


# ==============================================================================
# DeepFM Prediction Functions
# ==============================================================================

def load_deepfm_models(checkpoint_dir):
    """Load DeepFM K-Fold models from specific checkpoint"""
    print(f"\n[1/5] Loading DeepFM models from: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"DeepFM checkpoint not found: {checkpoint_dir}")

    models = []

    # Load all 5 fold models
    for fold in range(1, 6):
        model_path = os.path.join(checkpoint_dir, f"fold_{fold}_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fold {fold} model not found: {model_path}")

        print(f"   Loading fold {fold}...")
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

        processor = checkpoint['processor']
        config = checkpoint.get('config', {})

        # Initialize DynamicEmbeddingDeepFM model
        model = DynamicEmbeddingDeepFM(
            feature_dims=processor.categorical_feature_dims,
            hidden_units=config.get('HIDDEN_DIMS', [256, 128]),
            dropout=config.get('DROPOUT', 0.25),
            num_numerical=len(processor.numerical_features),
            min_embed_dim=config.get('MIN_EMBED_DIM', 4),
            max_embed_dim=config.get('MAX_EMBED_DIM', 50),
            use_dynamic_embedding=True
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        models.append(model)

    # Get processor from first checkpoint (all should be identical)
    first_checkpoint = torch.load(os.path.join(checkpoint_dir, "fold_1_model.pth"),
                                   weights_only=False, map_location='cpu')
    processor = first_checkpoint['processor']

    print(f"   Loaded {len(models)} DeepFM models")
    return models, processor


def predict_deepfm(models, processor, test_df, device, batch_size=10000):
    """Generate DeepFM K-Fold ensemble predictions"""
    print(f"\n[2/5] Generating DeepFM predictions...")

    # Process test data
    print("   Processing test data with DeepFM processor...")
    test_processed = processor.transform(test_df.copy())

    # Create batches
    num_batches = (len(test_processed) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(test_processed))
        batch_df = test_processed.iloc[start_idx:end_idx]

        # Categorical data
        categorical_data = {}
        for col in processor.categorical_features:
            if col in batch_df.columns:
                categorical_data[col] = torch.tensor(batch_df[col].values, dtype=torch.long)

        # Numerical data
        numerical_data = None
        if processor.numerical_features:
            numerical_cols = [col for col in processor.numerical_features if col in batch_df.columns]
            if numerical_cols:
                numerical_data = torch.tensor(batch_df[numerical_cols].values, dtype=torch.float32)

        batches.append((categorical_data, numerical_data))

    # Move models to device
    for model in models:
        model.to(device)
        model.eval()

    # Generate predictions
    all_predictions = []

    print(f"   Predicting with {len(models)} models on {len(batches)} batches...")
    with torch.no_grad():
        for categorical_data, numerical_data in tqdm(batches, desc="   DeepFM Prediction"):
            batch_predictions = []

            # Predict with each fold model
            for model in models:
                # Move batch to device
                batch_categorical = {}
                for key, data in categorical_data.items():
                    batch_categorical[key] = data.to(device)

                batch_numerical = None
                if numerical_data is not None:
                    batch_numerical = numerical_data.to(device)

                # Predict
                outputs = model(batch_categorical, batch_numerical)
                predictions = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                batch_predictions.append(predictions)

            # Average across folds
            ensemble_predictions = np.mean(batch_predictions, axis=0)
            all_predictions.append(ensemble_predictions)

    # Concatenate all batches
    final_predictions = np.concatenate(all_predictions)

    print(f"   DeepFM predictions: {len(final_predictions):,} samples")
    print(f"   Mean: {final_predictions.mean():.6f}, Std: {final_predictions.std():.6f}")

    return final_predictions


# ==============================================================================
# XGBoost Prediction Functions
# ==============================================================================

def load_xgboost_models(checkpoint_dir):
    """Load XGBoost K-Fold models from specific checkpoint"""
    print(f"\n[3/5] Loading XGBoost models from: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"XGBoost checkpoint not found: {checkpoint_dir}")

    import xgboost as xgb

    models = []
    model_files = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith("xgboost_fold_") and f.endswith(".json")]
    model_files.sort()

    for model_file in model_files:
        model_path = os.path.join(checkpoint_dir, model_file)
        model = xgb.Booster()
        model.load_model(model_path)
        models.append(model)

    print(f"   Loaded {len(models)} XGBoost models")
    return models


def predict_xgboost(models, test_df):
    """Generate XGBoost K-Fold ensemble predictions"""
    print(f"\n[4/5] Generating XGBoost predictions...")

    import xgboost as xgb

    # Load processor and feature names
    with open(XGBOOST_PROCESSOR_PATH, 'rb') as f:
        processor = pickle.load(f)

    with open(XGBOOST_FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)

    # Process test data
    print("   Processing test data with XGBoost processor...")
    test_processed = processor.transform(test_df.copy())
    test_processed = test_processed[feature_names]

    # Create DMatrix
    dtest = xgb.DMatrix(test_processed.values, feature_names=feature_names)

    # Generate predictions
    predictions = np.zeros(len(test_processed))

    print(f"   Predicting with {len(models)} models...")
    for i, model in enumerate(models, 1):
        fold_pred = model.predict(dtest)
        predictions += fold_pred / len(models)
        print(f"   Fold {i} done")

    # Clip predictions
    predictions = np.clip(predictions, 0, 1)
    predictions = np.nan_to_num(predictions, nan=0.5)

    print(f"   XGBoost predictions: {len(predictions):,} samples")
    print(f"   Mean: {predictions.mean():.6f}, Std: {predictions.std():.6f}")

    return predictions


# ==============================================================================
# Ensemble Functions
# ==============================================================================

def create_ensemble(xgb_predictions, deepfm_predictions, test_ids):
    """Create ensemble predictions with multiple weight combinations"""
    print(f"\n[5/5] Creating ensemble predictions...")

    # Weight combinations (XGBoost, DeepFM, name)
    weight_combinations = [
        (0.80, 0.20, "80_20"),
        (0.75, 0.25, "75_25"),
        (0.70, 0.30, "70_30"),
        (0.65, 0.35, "65_35"),
        (0.60, 0.40, "60_40"),
        (0.50, 0.50, "50_50"),
    ]

    ensemble_files = []

    for xgb_w, deepfm_w, name in weight_combinations:
        # Create ensemble
        ensemble_preds = xgb_w * xgb_predictions + deepfm_w * deepfm_predictions

        # Create submission dataframe
        result_df = pd.DataFrame({
            'ID': test_ids,
            'clicked': ensemble_preds
        })

        # Save
        filename = f"{OUTPUT_DIR}/ensemble_xgb{int(xgb_w*100)}_deepfm{int(deepfm_w*100)}_reproduction.csv"
        result_df.to_csv(filename, index=False)
        ensemble_files.append(filename)

        print(f"   {name}: Mean={ensemble_preds.mean():.6f} -> {os.path.basename(filename)}")

    return ensemble_files


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main inference function"""
    print("="*80)
    print("Toss CTR Prediction - Inference Script")
    print("Reproducing October 12, 2025 Final Submission")
    print("="*80)

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load test data
        print(f"\n[0/5] Loading test data from: {TEST_DATA_PATH}")
        test = pd.read_parquet(TEST_DATA_PATH, engine="pyarrow")
        test_ids = test['ID'].copy() if 'ID' in test.columns else None
        test_data = test.drop(columns=['ID']) if 'ID' in test.columns else test
        print(f"   Test data shape: {test_data.shape}")

        # === DeepFM Prediction ===
        deepfm_models, deepfm_processor = load_deepfm_models(DEEPFM_CHECKPOINT)
        deepfm_predictions = predict_deepfm(deepfm_models, deepfm_processor, test_data, device)

        # Save DeepFM predictions
        deepfm_file = f"{OUTPUT_DIR}/deepfm_reproduction.csv"
        pd.DataFrame({
            'ID': test_ids,
            'clicked': deepfm_predictions
        }).to_csv(deepfm_file, index=False)
        print(f"\n   Saved: {deepfm_file}")

        # === XGBoost Prediction ===
        xgb_models = load_xgboost_models(XGBOOST_CHECKPOINT)
        xgb_predictions = predict_xgboost(xgb_models, test_data)

        # Save XGBoost predictions
        xgb_file = f"{OUTPUT_DIR}/xgboost_reproduction.csv"
        pd.DataFrame({
            'ID': test_ids,
            'clicked': xgb_predictions
        }).to_csv(xgb_file, index=False)
        print(f"\n   Saved: {xgb_file}")

        # === Create Ensemble ===
        ensemble_files = create_ensemble(xgb_predictions, deepfm_predictions, test_ids)

        # === Summary ===
        print("\n" + "="*80)
        print("INFERENCE COMPLETE!")
        print("="*80)
        print(f"\nGenerated Files:")
        print(f"  XGBoost:  {xgb_file}")
        print(f"  DeepFM:   {deepfm_file}")
        print(f"  Ensemble: {len(ensemble_files)} files in {OUTPUT_DIR}/")
        print("\nCheckpoints Used:")
        print(f"  DeepFM:  {DEEPFM_CHECKPOINT}")
        print(f"  XGBoost: {XGBOOST_CHECKPOINT}")
        print("\nAll predictions saved successfully!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
