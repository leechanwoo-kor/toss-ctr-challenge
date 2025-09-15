"""
단순화된 훈련 파이프라인
NVIDIA Merlin 접근법 기반 최적 파라미터 사용
성능: 0.352247 ± 0.000230 (11.12% 향상)
"""
import os
import gc
import time
import numpy as np
import pandas as pd
import json
import xgboost as xgb
import wandb
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from utils import print_memory, clear_memory, calculate_competition_score
from config import (
    OUTPUT_X, OUTPUT_Y, MODEL_SHARED_DIR, RANDOM_STATE,
    get_device_config
)

# 훈련 전용 설정
RUN_TYPE = "train_simple"

def get_optimal_params(pos_ratio):
    """NVIDIA Merlin 최적 파라미터"""
    scale_pos_weight = (1 - pos_ratio) / pos_ratio

    # 디바이스 설정 가져오기
    device_config = get_device_config()

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

    return params

def train_simple_model():
    """단순화된 모델 훈련"""
    print("="*60)
    print("[SIMPLE_TRAINING] NVIDIA Merlin Style Training")
    print("="*60)
    print("Goal: Achieve CV Score ~0.352 with simple approach")
    print("Method: Use optimal parameters with 117 core features")

    start_time = time.time()
    print_memory("Initial")

    # 데이터 로드
    print("\n[DATA_LOADING] Loading processed data...")
    X_df = pd.read_parquet(OUTPUT_X)
    y = np.load(OUTPUT_Y)
    print(f"  Features shape: {X_df.shape}")
    print(f"  Target shape: {y.shape}")
    print_memory("After loading")

    # 데이터 분석
    pos_ratio = y.mean()
    print(f"\n[DATA_ANALYSIS] Dataset information:")
    print(f"  Samples: {X_df.shape[0]:,}")
    print(f"  Features: {X_df.shape[1]}")
    print(f"  Positive ratio: {pos_ratio:.4f}")

    # 최적 파라미터 설정
    params = get_optimal_params(pos_ratio)

    print(f"\n[MODEL_CONFIG] XGBoost parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # WandB 초기화
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"simple-train-{timestamp}"

    print(f"\n[WANDB] Initializing experiment tracking...")
    run = wandb.init(
        project="toss-ctr-challenge",
        name=run_name,
        config=params,
        job_type=RUN_TYPE,
        tags=["simple", "nvidia_merlin", "optimized"]
    )

    # Cross-validation 설정
    n_folds = 5
    wandb.config.update({
        "n_folds": n_folds,
        "features": X_df.shape[1],
        "samples": X_df.shape[0]
    })

    print(f"\n[CROSS_VALIDATION] {n_folds}-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = []
    cv_ap = []
    cv_wll = []
    cv_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), 1):
        fold_start = time.time()
        print(f"\n  Fold {fold}/{n_folds}:")
        print(f"    Train: {len(train_idx):,} | Val: {len(val_idx):,}")

        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
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
        cv_ap.append(ap)
        cv_wll.append(wll)
        cv_models.append(model)

        print(f"    Competition Score: {score:.6f}")
        print(f"    Average Precision: {ap:.6f}")
        print(f"    Weighted LogLoss: {wll:.6f}")
        print(f"    Best iteration: {model.best_iteration}")
        print(f"    Fold time: {time.time() - fold_start:.1f}s")

        # WandB 로깅
        wandb.log({
            "fold": fold,
            "competition_score": score,
            "average_precision": ap,
            "weighted_logloss": wll,
            "best_iteration": model.best_iteration,
            "fold_time": time.time() - fold_start
        })

        # 메모리 정리
        del X_train, X_val, y_train, y_val, dtrain, dval
        clear_memory()

    # 결과 분석
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    mean_ap = np.mean(cv_ap)
    mean_wll = np.mean(cv_wll)

    print(f"\n" + "="*60)
    print(f"[RESULTS] Cross-Validation Results")
    print(f"="*60)
    print(f"Competition Score: {mean_score:.6f} ± {std_score:.6f}")
    print(f"Average Precision: {mean_ap:.6f}")
    print(f"Weighted LogLoss: {mean_wll:.6f}")
    print(f"All fold scores: {[f'{s:.6f}' for s in cv_scores]}")

    # WandB 요약 통계
    wandb.summary.update({
        "cv_score_mean": mean_score,
        "cv_score_std": std_score,
        "cv_ap_mean": mean_ap,
        "cv_wll_mean": mean_wll,
        "cv_scores": cv_scores
    })

    # 성능 분석
    target_score = 0.353
    baseline_score = 0.317

    print(f"\n[PERFORMANCE_ANALYSIS] Performance comparison:")
    print(f"  Baseline (complex): {baseline_score:.6f}")
    print(f"  Simple training: {mean_score:.6f}")
    print(f"  NVIDIA target: {target_score:.6f}")

    if mean_score > baseline_score:
        improvement = ((mean_score - baseline_score) / baseline_score) * 100
        print(f"  [SUCCESS] Improved by {improvement:+.2f}%")

        if mean_score >= target_score:
            print(f"  [EXCELLENT] Target achieved!")
        else:
            gap = target_score - mean_score
            print(f"  [GOOD] Gap to target: {gap:.6f}")
    else:
        decline = ((baseline_score - mean_score) / baseline_score) * 100
        print(f"  [WARNING] Declined by {decline:.2f}%")

    # 최고 성능 모델 저장
    best_fold_idx = np.argmax(cv_scores)
    best_model = cv_models[best_fold_idx]

    print(f"\n[MODEL_SAVING] Saving best model...")
    print(f"  Best fold: {best_fold_idx + 1}")
    print(f"  Best score: {cv_scores[best_fold_idx]:.6f}")

    # 모델 저장 경로 설정
    MODEL_SHARED_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    model_name = f'simple_model_{timestamp}_cv{mean_score:.4f}.json'
    model_path = MODEL_SHARED_DIR / model_name

    best_model.save_model(str(model_path))
    print(f"  Model saved to: {model_path}")

    # 결과 저장
    results = {
        'cv_score_mean': float(mean_score),
        'cv_score_std': float(std_score),
        'cv_scores': [float(s) for s in cv_scores],
        'cv_ap_mean': float(mean_ap),
        'cv_wll_mean': float(mean_wll),
        'best_fold': int(best_fold_idx + 1),
        'params': params,
        'approach': 'nvidia_merlin_simple',
        'features_used': X_df.shape[1],
        'training_time': time.time() - start_time
    }

    results_path = MODEL_SHARED_DIR / f'simple_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    print(f"\n[TIMING] Training completed in {time.time() - start_time:.1f}s")

    # WandB에 모델 아티팩트 추가
    try:
        artifact = wandb.Artifact(f"simple_model_{timestamp}", type='model')
        artifact.add_file(str(model_path))
        artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
        print(f"  Model uploaded to WandB")
    except Exception as e:
        print(f"  Warning: Could not upload to WandB: {e}")

    # WandB 종료
    wandb.finish()

    return best_model, mean_score, results

def main():
    """메인 실행 함수"""
    model, score, results = train_simple_model()

    print("\n" + "="*60)
    print("[SUCCESS] Simple training completed!")
    print(f"Final CV Score: {score:.6f}")
    print("Model ready for inference!")
    print("="*60)

    return model, score, results

if __name__ == '__main__':
    main()