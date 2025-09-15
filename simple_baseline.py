"""
최적화된 Toss CTR 예측 모델 - 단순 베이스라인
NVIDIA Merlin 접근법 기반: 117개 핵심 피처만 사용하여 최고 성능 달성
CV Score: 0.352247 ± 0.000230 (11.12% 성능 향상)
"""
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
from sklearn.model_selection import StratifiedKFold

# Add src to path for imports
sys.path.append('src')

from utils import calculate_competition_score, print_memory, clear_memory
from config import (
    OUTPUT_X, OUTPUT_Y, MODEL_SHARED_DIR, RANDOM_STATE,
    CATEGORICAL_FEATURES, get_device_config
)

def get_core_features():
    """NVIDIA Merlin 방식: 117개 핵심 피처만 사용"""

    # Use categorical features from config
    categorical_features = CATEGORICAL_FEATURES.copy()

    # CONTINUOUS COLUMNS (112개 - 원본 피처들만)
    continuous_features = (
        [f'feat_a_{i}' for i in range(1, 19)] +  # 18
        [f'feat_b_{i}' for i in range(1, 7)] +   # 6
        [f'feat_c_{i}' for i in range(1, 9)] +   # 8
        [f'feat_d_{i}' for i in range(1, 7)] +   # 6
        [f'feat_e_{i}' for i in range(1, 11)] +  # 10
        [f'history_a_{i}' for i in range(1, 8)] +  # 7
        [f'history_b_{i}' for i in range(1, 31)] + # 30
        [f'l_feat_{i}' for i in range(1, 28)]      # 27
    )

    core_features = categorical_features + continuous_features

    print(f"[FEATURE_SELECTION] Core features for optimal performance:")
    print(f"  Categorical: {len(categorical_features)} features")
    print(f"  Continuous: {len(continuous_features)} features")
    print(f"  Total: {len(core_features)} features (147 → 117, -30)")

    return core_features

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print(f"[DATA_LOADING] Loading processed data...")
    print_memory("Before loading")

    X_full = pd.read_parquet(OUTPUT_X)
    y = np.load(OUTPUT_Y)

    print(f"  Original shape: {X_full.shape}")

    # 핵심 피처만 선택
    core_features = get_core_features()
    X_core = X_full[core_features].copy()

    print(f"  Core features shape: {X_core.shape}")
    print(f"  Removed features: {X_full.shape[1] - X_core.shape[1]}")

    # 메모리 정리
    del X_full
    clear_memory()

    # 결측값 처리 (NVIDIA Merlin 방식: 0으로 채우기)
    missing_count = X_core.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Filling {missing_count:,} missing values with 0")
        X_core = X_core.fillna(0)

    print_memory("After preprocessing")
    return X_core, y

def run_simple_baseline():
    """단순 베이스라인 실행"""
    print("="*60)
    print("[SIMPLE_BASELINE] NVIDIA Merlin Style Baseline")
    print("="*60)
    print("Goal: Recover performance from 0.317 to 0.353+")
    print("Method: Use only 117 core features (no seq/eng features)")

    start_time = time.time()

    # 데이터 로드 및 전처리
    X, y = load_and_prepare_data()

    # 데이터 분석
    pos_ratio = y.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    print(f"\n[DATA_ANALYSIS] Dataset information:")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive ratio: {pos_ratio:.4f}")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    # 디바이스 설정 가져오기
    device_config = get_device_config()

    # NVIDIA Merlin 최적 파라미터
    params = {
        'objective': 'binary:logistic',
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': 0,
        'seed': RANDOM_STATE,
        **device_config  # GPU/CPU 자동 감지 설정
    }

    print(f"\n[MODEL_CONFIG] XGBoost parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 3-fold CV로 빠른 검증
    print(f"\n[CROSS_VALIDATION] 3-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/3:")
        print(f"    Train: {len(train_idx):,} | Val: {len(val_idx):,}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        y_pred = model.predict(dval)
        score, ap, wll = calculate_competition_score(y_val, y_pred)
        cv_scores.append(score)

        print(f"    Competition Score: {score:.6f}")
        print(f"    Average Precision: {ap:.6f}")
        print(f"    Weighted LogLoss: {wll:.6f}")
        print(f"    Best iteration: {model.best_iteration}")

        # 메모리 정리
        del X_train, X_val, y_train, y_val, dtrain, dval, model
        clear_memory()

    # 결과 분석
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    print(f"\n" + "="*60)
    print(f"[RESULTS] Simple Baseline Results")
    print(f"="*60)
    print(f"CV Score: {mean_score:.6f} ± {std_score:.6f}")
    print(f"All fold scores: {[f'{s:.6f}' for s in cv_scores]}")

    # 성능 비교
    target_score = 0.353
    current_score = 0.317

    print(f"\n[PERFORMANCE_ANALYSIS] Performance comparison:")
    print(f"  Previous (complex): {current_score:.6f}")
    print(f"  Simple baseline: {mean_score:.6f}")
    print(f"  NVIDIA target: {target_score:.6f}")

    if mean_score > current_score:
        improvement = ((mean_score - current_score) / current_score) * 100
        print(f"  [SUCCESS] Improved by {improvement:+.2f}%")

        if mean_score >= target_score:
            print(f"  [EXCELLENT] Target achieved!")
        else:
            gap = target_score - mean_score
            print(f"  [GOOD] Gap to target: {gap:.6f}")
    else:
        decline = ((current_score - mean_score) / current_score) * 100
        print(f"  [WARNING] Declined by {decline:.2f}%")

    print(f"\n[TIMING] Completed in {time.time() - start_time:.1f}s")

    # 결과 저장
    if mean_score > current_score:
        # 결과 저장을 위한 디렉토리 생성
        MODEL_SHARED_DIR.mkdir(parents=True, exist_ok=True)
        simple_params_path = MODEL_SHARED_DIR / 'simple_baseline_params.json'

        result_data = {
            'cv_score': float(mean_score),
            'cv_std': float(std_score),
            'cv_scores': [float(s) for s in cv_scores],
            'params': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in params.items()},
            'features_used': len(get_core_features()),
            'approach': 'nvidia_merlin_simple',
            'improvement_pct': ((mean_score - current_score) / current_score) * 100
        }

        with open(simple_params_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"\n[SAVED] Simple baseline results saved to: {simple_params_path}")

    return cv_scores

if __name__ == "__main__":
    run_simple_baseline()