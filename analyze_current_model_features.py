"""
최신 모델의 피처 구성과 중요도 분석
"""
import json
import pickle
import pandas as pd
import xgboost as xgb
from collections import defaultdict

def analyze_current_model():
    """최신 모델의 피처 분석"""

    # 피처 이름 로드
    with open('data/processed/xgboost_enhanced/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    print(f"Total features: {len(feature_names)}")
    print(f"First 10 features: {feature_names[:10]}")
    print(f"Last 10 features: {feature_names[-10:]}")

    # 최신 모델 체크포인트 로드
    model_dir = 'checkpoints/xgboost/xgboost-kfold-0919_1007'

    # 모든 fold 모델의 피처 중요도 수집
    all_importance = defaultdict(list)

    for fold in range(1, 6):  # fold 1-5
        model_path = f"{model_dir}/xgboost_fold_{fold}.json"

        try:
            model = xgb.Booster()
            model.load_model(model_path)

            # Gain 중요도 추출
            importance = model.get_score(importance_type='gain')

            # 각 피처의 중요도를 기록
            for feature_name in feature_names:
                gain_value = importance.get(feature_name, 0.0)
                all_importance[feature_name].append(gain_value)

            print(f"Fold {fold} loaded - {len(importance)} feature importance extracted")

        except Exception as e:
            print(f"Fold {fold} failed to load: {e}")

    # 평균 중요도 계산
    avg_importance = {}
    for feature_name, gains in all_importance.items():
        if gains:  # 값이 있는 경우만
            avg_importance[feature_name] = sum(gains) / len(gains)
        else:
            avg_importance[feature_name] = 0.0

    # 중요도 순으로 정렬
    sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    # 결과 출력
    print(f"\nTop 20 feature importance:")
    for i, (feature, importance) in enumerate(sorted_importance[:20], 1):
        print(f"{i:2d}. {feature:<25} : {importance:>8.2f}")

    print(f"\nBottom 20 feature importance:")
    for i, (feature, importance) in enumerate(sorted_importance[-20:], len(sorted_importance)-19):
        print(f"{i:2d}. {feature:<25} : {importance:>8.2f}")

    # Feature group analysis
    feature_groups = defaultdict(list)
    for feature, importance in sorted_importance:
        if feature.startswith('feat_a_'):
            feature_groups['feat_a'].append((feature, importance))
        elif feature.startswith('feat_b_'):
            feature_groups['feat_b'].append((feature, importance))
        elif feature.startswith('feat_c_'):
            feature_groups['feat_c'].append((feature, importance))
        elif feature.startswith('feat_d_'):
            feature_groups['feat_d'].append((feature, importance))
        elif feature.startswith('feat_e_'):
            feature_groups['feat_e'].append((feature, importance))
        elif feature.startswith('history_a_'):
            feature_groups['history_a'].append((feature, importance))
        elif feature.startswith('history_b_'):
            feature_groups['history_b'].append((feature, importance))
        elif feature.startswith('l_feat_'):
            feature_groups['l_feat'].append((feature, importance))
        else:
            feature_groups['other'].append((feature, importance))

    print(f"\nFeature group analysis:")
    for group, features in feature_groups.items():
        if features:
            total_importance = sum(imp for _, imp in features)
            avg_importance = total_importance / len(features)
            max_importance = max(imp for _, imp in features)
            print(f"{group:<12}: {len(features):2d} features | Total:{total_importance:8.1f} | Avg:{avg_importance:6.1f} | Max:{max_importance:6.1f}")

    # Save to CSV
    df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
    df['Rank'] = range(1, len(df) + 1)
    df.to_csv('current_model_feature_importance.csv', index=False)
    print(f"\nFeature importance saved to 'current_model_feature_importance.csv'")

    # Check zero importance features
    zero_importance = [f for f, imp in sorted_importance if imp == 0.0]
    if zero_importance:
        print(f"\nZero importance features ({len(zero_importance)} total):")
        for feat in zero_importance[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(zero_importance) > 10:
            print(f"  ... and {len(zero_importance)-10} more")

    return sorted_importance, feature_groups

if __name__ == "__main__":
    analyze_current_model()