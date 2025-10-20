"""
XGBoost Feature Importance 분석 도구
- 학습된 K-Fold 모델에서 feature importance 추출
- 전체 fold 평균 importance 계산
- 시각화 및 상위/하위 피처 분석
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_kfold_models(checkpoint_dir):
    """K-Fold 모델들 로드"""
    print(f"📁 Loading K-Fold models from {checkpoint_dir}")

    models = []
    fold_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('xgboost_fold_') and f.endswith('.json')])

    if not fold_files:
        raise FileNotFoundError(f"No model files found in {checkpoint_dir}")

    for fold_file in fold_files:
        model_path = os.path.join(checkpoint_dir, fold_file)
        model = xgb.Booster()
        model.load_model(model_path)
        models.append(model)
        print(f"   ✓ Loaded {fold_file}")

    print(f"   Total models loaded: {len(models)}")
    return models


def extract_feature_importance(models, importance_type='gain'):
    """
    모델들로부터 feature importance 추출

    Args:
        models: XGBoost 모델 리스트
        importance_type: 'gain', 'weight', 'cover' 중 선택
            - gain: 피처가 split에 기여한 평균 gain (정확도 개선)
            - weight: 피처가 사용된 횟수
            - cover: 피처가 커버하는 샘플 수
    """
    print(f"\n📊 Extracting feature importance (type: {importance_type})...")

    all_importance = []

    for i, model in enumerate(models, 1):
        importance = model.get_score(importance_type=importance_type)
        all_importance.append(importance)
        print(f"   Fold {i}: {len(importance)} features used")

    # 전체 피처 집합
    all_features = set()
    for imp in all_importance:
        all_features.update(imp.keys())

    print(f"   Total unique features: {len(all_features)}")

    # 평균 importance 계산
    avg_importance = {}
    for feature in all_features:
        scores = [imp.get(feature, 0) for imp in all_importance]
        avg_importance[feature] = np.mean(scores)

    # DataFrame으로 변환 (정렬)
    importance_df = pd.DataFrame([
        {'feature': feature, 'importance': score}
        for feature, score in avg_importance.items()
    ]).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


def analyze_importance_distribution(importance_df):
    """Importance 분포 분석"""
    print("\n📈 Feature Importance Distribution Analysis:")
    print("=" * 70)

    total_importance = importance_df['importance'].sum()

    # 상위 k% 피처가 전체 importance의 몇 %를 차지하는지
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        n_features = int(len(importance_df) * pct / 100)
        top_importance = importance_df.head(n_features)['importance'].sum()
        contribution = (top_importance / total_importance) * 100
        print(f"   Top {pct:2d}% features ({n_features:3d} features): {contribution:5.2f}% of total importance")

    print("=" * 70)

    # 상위/하위 피처 요약
    print(f"\n🔝 Top 20 Most Important Features:")
    for idx, row in importance_df.head(20).iterrows():
        print(f"   {idx+1:2d}. {row['feature']:30s}: {row['importance']:10.2f}")

    print(f"\n🔻 Bottom 20 Least Important Features:")
    for idx, row in importance_df.tail(20).iterrows():
        print(f"   {idx+1:2d}. {row['feature']:30s}: {row['importance']:10.2f}")


def plot_importance_distribution(importance_df, save_path):
    """Feature importance 시각화"""
    print(f"\n📊 Creating importance visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top 30 features
    top_30 = importance_df.head(30)
    axes[0, 0].barh(range(len(top_30)), top_30['importance'].values)
    axes[0, 0].set_yticks(range(len(top_30)))
    axes[0, 0].set_yticklabels(top_30['feature'].values, fontsize=8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel('Importance (Gain)')
    axes[0, 0].set_title('Top 30 Most Important Features')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # 2. Cumulative importance
    cumsum = importance_df['importance'].cumsum() / importance_df['importance'].sum() * 100
    axes[0, 1].plot(range(len(cumsum)), cumsum.values, linewidth=2)
    axes[0, 1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
    axes[0, 1].axhline(y=90, color='orange', linestyle='--', label='90% threshold')
    axes[0, 1].axhline(y=95, color='green', linestyle='--', label='95% threshold')
    axes[0, 1].set_xlabel('Number of Features (sorted by importance)')
    axes[0, 1].set_ylabel('Cumulative Importance (%)')
    axes[0, 1].set_title('Cumulative Feature Importance')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 80%, 90%, 95% 달성 피처 수 표시
    for threshold in [80, 90, 95]:
        n_features = (cumsum >= threshold).idxmax()
        axes[0, 1].scatter([n_features], [threshold], s=100, zorder=5)
        axes[0, 1].annotate(f'{n_features} features\n({threshold}%)',
                           xy=(n_features, threshold),
                           xytext=(10, -10), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Importance distribution (histogram)
    axes[1, 0].hist(importance_df['importance'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Importance Value')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Distribution of Feature Importance Values')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)

    # 4. Top/Bottom comparison (percentage)
    percentages = [10, 20, 30, 40, 50]
    top_contributions = []
    bottom_contributions = []
    total_imp = importance_df['importance'].sum()

    for pct in percentages:
        n = int(len(importance_df) * pct / 100)
        top_contributions.append(importance_df.head(n)['importance'].sum() / total_imp * 100)
        bottom_contributions.append(importance_df.tail(n)['importance'].sum() / total_imp * 100)

    x = np.arange(len(percentages))
    width = 0.35
    axes[1, 1].bar(x - width/2, top_contributions, width, label='Top k% features', alpha=0.8)
    axes[1, 1].bar(x + width/2, bottom_contributions, width, label='Bottom k% features', alpha=0.8)
    axes[1, 1].set_xlabel('Percentage of Features')
    axes[1, 1].set_ylabel('Contribution to Total Importance (%)')
    axes[1, 1].set_title('Top vs Bottom Features Importance Contribution')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{p}%' for p in percentages])
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization: {save_path}")
    plt.close()


def suggest_feature_selection_thresholds(importance_df):
    """Feature selection threshold 제안"""
    print("\n💡 Feature Selection Threshold Suggestions:")
    print("=" * 70)

    total_imp = importance_df['importance'].sum()
    cumsum = importance_df['importance'].cumsum() / total_imp * 100

    suggestions = []

    # 80%, 90%, 95% importance를 커버하는 피처 수
    for threshold in [80, 90, 95]:
        n_features = (cumsum >= threshold).idxmax()
        pct_features = (n_features / len(importance_df)) * 100
        suggestions.append({
            'threshold': threshold,
            'n_features': n_features,
            'pct_features': pct_features
        })
        print(f"   {threshold}% importance: {n_features} features ({pct_features:.1f}% of total)")

    print("=" * 70)
    print("\n📌 Recommendations based on your graph analysis:")
    print("   - Graph shows performance degradation with removed features")
    print("   - Suggest starting with 70-80% feature retention")
    print("   - This corresponds to removing bottom 20-30% features")

    # 하위 20-30% 피처 수
    for remove_pct in [20, 30, 40, 50]:
        n_remove = int(len(importance_df) * remove_pct / 100)
        n_keep = len(importance_df) - n_remove
        keep_imp = importance_df.head(n_keep)['importance'].sum() / total_imp * 100
        print(f"   - Remove bottom {remove_pct}% ({n_remove} features) → Keep {n_keep} features ({keep_imp:.1f}% importance)")

    return suggestions


def main():
    """메인 실행 함수"""
    print("🔍 XGBoost Feature Importance Analysis")
    print("=" * 70)

    # 최신 체크포인트 찾기 (또는 수동 지정)
    checkpoint_base = "checkpoints/xgboost"

    # 최신 checkpoint 자동 선택
    all_checkpoints = [d for d in os.listdir(checkpoint_base) if os.path.isdir(os.path.join(checkpoint_base, d))]
    all_checkpoints = [d for d in all_checkpoints if d.startswith('xgboost-')]
    all_checkpoints.sort(reverse=True)

    if not all_checkpoints:
        print("❌ No checkpoints found!")
        return

    latest_checkpoint = all_checkpoints[0]
    checkpoint_dir = os.path.join(checkpoint_base, latest_checkpoint)

    print(f"📂 Using checkpoint: {latest_checkpoint}")
    print(f"   Path: {checkpoint_dir}")

    # 모델 로드
    try:
        models = load_kfold_models(checkpoint_dir)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Feature importance 추출 (gain 기준)
    importance_df = extract_feature_importance(models, importance_type='gain')

    # 분석
    analyze_importance_distribution(importance_df)

    # 시각화
    output_dir = os.path.join(checkpoint_dir, "feature_analysis")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "feature_importance_analysis.png")
    plot_importance_distribution(importance_df, plot_path)

    # CSV 저장
    csv_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved importance data: {csv_path}")

    # Feature selection 제안
    suggestions = suggest_feature_selection_thresholds(importance_df)

    # 제안 저장
    suggestions_df = pd.DataFrame(suggestions)
    suggestions_path = os.path.join(output_dir, "feature_selection_suggestions.csv")
    suggestions_df.to_csv(suggestions_path, index=False)
    print(f"💾 Saved suggestions: {suggestions_path}")

    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print(f"📁 Results saved in: {output_dir}")
    print("=" * 70)

    # 다음 단계 안내
    print("\n🚀 Next Steps:")
    print("   1. Review the visualization and CSV file")
    print("   2. Choose a feature selection threshold (recommended: 70-80% retention)")
    print("   3. Run training with selected features using:")
    print("      python -m src.xgboost.xgboost_train_with_selection --threshold 0.7")


if __name__ == "__main__":
    main()
