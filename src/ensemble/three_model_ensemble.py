"""
XGBoost v4 + XGBoost v5 + DeepFM v2.2 - 3-Model Ensemble
다양성 극대화 전략: 서로 다른 특성을 가진 3개 모델 결합
"""
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    print("=== 3-Model Ensemble: XGBoost v4 + v5 + DeepFM v2.2 ===")
    print()

    # 파일 경로
    xgb_v4_file = "submissions/xgboost/xgboost_kfold_1007_2346.csv"  # v4: LB 0.348995
    xgb_v5_file = "submissions/xgboost/xgboost_kfold_1009_1255.csv"  # v5: LB 0.348981
    deepfm_file = "submissions/deepfm/deepfm_kfold_ensemble_20251005_004836.csv"  # v2.2: LB 0.3466

    # 실제 Public LB 점수
    xgb_v4_lb = 0.348995
    xgb_v5_lb = 0.348981
    deepfm_lb = 0.3466

    # 데이터 로드
    print("📂 Loading predictions...")
    xgb_v4_df = pd.read_csv(xgb_v4_file)
    xgb_v5_df = pd.read_csv(xgb_v5_file)
    deepfm_df = pd.read_csv(deepfm_file)

    print(f"   XGBoost v4: {len(xgb_v4_df):,} predictions (LB: {xgb_v4_lb})")
    print(f"   XGBoost v5: {len(xgb_v5_df):,} predictions (LB: {xgb_v5_lb})")
    print(f"   DeepFM v2.2: {len(deepfm_df):,} predictions (LB: {deepfm_lb})")
    print()

    # ID 정렬 확인
    xgb_v4_df = xgb_v4_df.sort_values('ID').reset_index(drop=True)
    xgb_v5_df = xgb_v5_df.sort_values('ID').reset_index(drop=True)
    deepfm_df = deepfm_df.sort_values('ID').reset_index(drop=True)

    # 3모델 가중치 조합들
    # 전략: XGBoost v4(최고성능) + v5(일반화) + DeepFM(다양성)
    weight_combinations = [
        # v4 중심 (v4가 가장 높은 LB)
        (0.45, 0.25, 0.30, "45_25_30"),  # v4 중심 + v5 일반화 + DeepFM 다양성
        (0.40, 0.30, 0.30, "40_30_30"),  # 균형잡힌 조합
        (0.50, 0.20, 0.30, "50_20_30"),  # v4 강조

        # v5 일반화 강조
        (0.35, 0.35, 0.30, "35_35_30"),  # v4-v5 동등 + DeepFM
        (0.40, 0.35, 0.25, "40_35_25"),  # XGBoost 중심

        # DeepFM 다양성 강조
        (0.40, 0.25, 0.35, "40_25_35"),  # DeepFM 강조
        (0.35, 0.30, 0.35, "35_30_35"),  # 균형

        # 탐색적 조합
        (0.45, 0.30, 0.25, "45_30_25"),
        (0.42, 0.28, 0.30, "42_28_30"),
        (0.38, 0.32, 0.30, "38_32_30"),
    ]

    timestamp = datetime.now().strftime('%m%d_%H%M')
    results = []

    print("🔄 Generating ensemble combinations...")
    print()

    # 각 조합별 앙상블 생성
    for v4_w, v5_w, deepfm_w, name in weight_combinations:
        # 가중치 합이 1인지 확인
        assert abs(v4_w + v5_w + deepfm_w - 1.0) < 1e-6, f"Weights must sum to 1.0: {v4_w + v5_w + deepfm_w}"

        # 앙상블 예측
        ensemble_preds = (v4_w * xgb_v4_df['clicked'] +
                         v5_w * xgb_v5_df['clicked'] +
                         deepfm_w * deepfm_df['clicked'])

        result_df = xgb_v4_df[['ID']].copy()
        result_df['clicked'] = ensemble_preds

        # 예상 Public LB 점수 계산
        expected_lb = v4_w * xgb_v4_lb + v5_w * xgb_v5_lb + deepfm_w * deepfm_lb

        filename = f"ensemble_3model_{name}_{timestamp}.csv"
        result_df.to_csv(f"submissions/ensemble/{filename}", index=False)

        results.append({
            'name': name,
            'v4_weight': v4_w,
            'v5_weight': v5_w,
            'deepfm_weight': deepfm_w,
            'expected_lb': expected_lb,
            'filename': filename,
            'mean': ensemble_preds.mean(),
            'std': ensemble_preds.std(),
            'min': ensemble_preds.min(),
            'max': ensemble_preds.max()
        })

        print(f"✅ {name}: Expected LB = {expected_lb:.6f}")
        print(f"   Weights: XGBv4 {v4_w:.0%} + XGBv5 {v5_w:.0%} + DeepFM {deepfm_w:.0%}")
        print(f"   File: {filename}")
        print(f"   Stats: mean={ensemble_preds.mean():.4f}, std={ensemble_preds.std():.4f}")
        print()

    # 결과 요약
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('expected_lb', ascending=False)

    print("=" * 80)
    print("📊 Top 5 Expected Scores:")
    print("=" * 80)
    for idx, row in results_df.head(5).iterrows():
        print(f"{row['name']}: {row['expected_lb']:.6f}")
        print(f"   → XGBv4 {row['v4_weight']:.0%} + XGBv5 {row['v5_weight']:.0%} + DeepFM {row['deepfm_weight']:.0%}")

    print()
    print(f"🎯 Best Expected: {results_df.iloc[0]['expected_lb']:.6f} ({results_df.iloc[0]['name']})")
    print(f"📁 All ensembles saved to: submissions/ensemble/")
    print()

    # 현재 최고 기록과 비교
    current_best = 0.349652  # XGBv4 + DeepFM (65:35)
    best_expected = results_df.iloc[0]['expected_lb']
    improvement = best_expected - current_best

    print(f"📈 Current Best: {current_best:.6f} (XGBv4 65% + DeepFM 35%)")
    print(f"🚀 New Expected: {best_expected:.6f}")
    print(f"💡 Expected Improvement: {improvement:+.6f}")

    if improvement > 0:
        print(f"✨ 3-model ensemble shows {improvement:.6f} expected improvement!")
    else:
        print(f"⚠️  Expected improvement is minimal ({improvement:.6f})")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
