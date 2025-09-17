"""
XGBoost + DeepFM 앙상블 예측 스크립트
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def create_ensemble_submission(
    xgb_file=None,
    deepfm_file=None,
    weights=None,
    output_suffix="ensemble"
):
    """
    XGBoost와 DeepFM 예측을 앙상블하여 최종 submission 생성

    Args:
        xgb_file: XGBoost 예측 파일 경로
        deepfm_file: DeepFM 예측 파일 경로
        weights: [xgb_weight, deepfm_weight] 가중치
        output_suffix: 출력 파일 접미사
    """

    print("🎯 Starting XGBoost + DeepFM Ensemble...")

    # 기본값 설정
    if xgb_file is None:
        xgb_file = "submissions/xgboost_xgboost-0917_1403_0917_1438.csv"  # 0.3366 모델 (checkpoints에서 생성)
    if deepfm_file is None:
        deepfm_file = "submissions/deepfm_deepfm-fast-0917_1110_0917_1212.csv"  # 최신 DeepFM
    if weights is None:
        weights = [0.8, 0.2]  # XGBoost 더 높은 가중치

    print(f"📊 XGBoost file: {xgb_file}")
    print(f"🧠 DeepFM file: {deepfm_file}")
    print(f"⚖️  Weights: XGBoost={weights[0]:.1f}, DeepFM={weights[1]:.1f}")

    # 파일 존재 확인
    if not os.path.exists(xgb_file):
        raise FileNotFoundError(f"XGBoost file not found: {xgb_file}")
    if not os.path.exists(deepfm_file):
        raise FileNotFoundError(f"DeepFM file not found: {deepfm_file}")

    # 예측 파일 로드
    print("📁 Loading prediction files...")
    xgb_df = pd.read_csv(xgb_file)
    deepfm_df = pd.read_csv(deepfm_file)

    print(f"XGBoost predictions shape: {xgb_df.shape}")
    print(f"DeepFM predictions shape: {deepfm_df.shape}")

    # ID 정렬 확인
    if not xgb_df['ID'].equals(deepfm_df['ID']):
        print("⚠️  ID mismatch detected, sorting both files...")
        xgb_df = xgb_df.sort_values('ID').reset_index(drop=True)
        deepfm_df = deepfm_df.sort_values('ID').reset_index(drop=True)

        if not xgb_df['ID'].equals(deepfm_df['ID']):
            raise ValueError("ID columns do not match even after sorting!")

    print("✅ ID columns verified")

    # 예측값 추출
    xgb_preds = xgb_df['clicked'].values
    deepfm_preds = deepfm_df['clicked'].values

    # 앙상블 예측 계산
    print("🔮 Computing ensemble predictions...")
    ensemble_preds = weights[0] * xgb_preds + weights[1] * deepfm_preds

    # 예측 범위 검증
    print("🔍 Validating ensemble predictions...")
    if np.any((ensemble_preds < 0) | (ensemble_preds > 1)):
        invalid_count = np.sum((ensemble_preds < 0) | (ensemble_preds > 1))
        print(f"⚠️  Warning: {invalid_count} predictions outside [0,1] range")
        ensemble_preds = np.clip(ensemble_preds, 0, 1)
        print("   Clipped to valid range")

    # NaN/Inf 검증
    if np.any(~np.isfinite(ensemble_preds)):
        nan_count = np.sum(~np.isfinite(ensemble_preds))
        print(f"⚠️  Warning: {nan_count} NaN/Inf predictions found")
        ensemble_preds = np.nan_to_num(ensemble_preds, nan=0.5, posinf=1.0, neginf=0.0)
        print("   Replaced with safe values")

    print("✅ Validation complete")

    # 앙상블 결과 생성
    ensemble_df = xgb_df[['ID']].copy()
    ensemble_df['clicked'] = ensemble_preds

    # 출력 파일 경로 생성
    timestamp = datetime.now().strftime('%m%d_%H%M')
    output_path = f"submissions/ensemble_{output_suffix}_{timestamp}.csv"

    # 파일 저장
    ensemble_df.to_csv(output_path, index=False)

    # 통계 정보 출력
    print("\n" + "="*60)
    print("🎉 Ensemble Submission Created!")
    print("="*60)

    print(f"\n📊 Individual Model Stats:")
    print(f"XGBoost - Min: {xgb_preds.min():.6f}, Max: {xgb_preds.max():.6f}, Mean: {xgb_preds.mean():.6f}")
    print(f"DeepFM  - Min: {deepfm_preds.min():.6f}, Max: {deepfm_preds.max():.6f}, Mean: {deepfm_preds.mean():.6f}")

    print(f"\n🎯 Ensemble Stats:")
    print(f"Count: {len(ensemble_preds):,}")
    print(f"Min: {ensemble_preds.min():.6f}")
    print(f"Max: {ensemble_preds.max():.6f}")
    print(f"Mean: {ensemble_preds.mean():.6f}")
    print(f"Std: {ensemble_preds.std():.6f}")

    print(f"\n💾 Saved to: {output_path}")

    # 예상 성능 추정
    xgb_score = 0.3366  # 실제 리더보드 점수
    deepfm_score = 0.317  # 추정 점수
    estimated_score = weights[0] * xgb_score + weights[1] * deepfm_score

    print(f"\n📈 Estimated Performance:")
    print(f"XGBoost Score: {xgb_score:.4f} (weight: {weights[0]:.1f})")
    print(f"DeepFM Score: {deepfm_score:.4f} (weight: {weights[1]:.1f})")
    print(f"Estimated Ensemble Score: {estimated_score:.4f}")

    if estimated_score >= 0.35:
        print("🎯 TARGET ACHIEVED: 0.35+ Competition Score!")
    else:
        print(f"🎯 Target: 0.35+ (estimated: {estimated_score:.4f})")

    return ensemble_preds, output_path


def try_multiple_weights():
    """여러 가중치 조합으로 앙상블 실험"""

    weight_combinations = [
        [0.9, 0.1],   # XGBoost 우세
        [0.8, 0.2],   # 기본 조합
        [0.7, 0.3],   # 균형
        [0.75, 0.25], # 중간값
        [0.85, 0.15], # XGBoost 강조
    ]

    results = []

    for i, weights in enumerate(weight_combinations):
        print(f"\n🔄 Testing weights combination {i+1}: {weights}")

        preds, output_path = create_ensemble_submission(
            weights=weights,
            output_suffix=f"w{int(weights[0]*100)}_{int(weights[1]*100)}"
        )

        # 예상 점수 계산
        estimated_score = weights[0] * 0.3366 + weights[1] * 0.317

        results.append({
            'weights': weights,
            'estimated_score': estimated_score,
            'output_path': output_path,
            'mean_pred': preds.mean()
        })

    # 결과 요약
    print("\n" + "="*80)
    print("📊 ENSEMBLE EXPERIMENT SUMMARY")
    print("="*80)

    for i, result in enumerate(results):
        print(f"{i+1}. Weights {result['weights']} → Score: {result['estimated_score']:.4f}, Mean: {result['mean_pred']:.4f}")

    # 최고 점수 찾기
    best_result = max(results, key=lambda x: x['estimated_score'])
    print(f"\n🏆 Best combination: {best_result['weights']} (Score: {best_result['estimated_score']:.4f})")
    print(f"📁 Best file: {best_result['output_path']}")

    return results


if __name__ == "__main__":
    # 기본 앙상블 생성
    try:
        create_ensemble_submission()
        print("\n🚀 Success! Ready for submission!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()