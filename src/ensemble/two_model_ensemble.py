"""
XGBoost + DeepFM K-Fold 2-Model Ensemble
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def main():
    # 파일 경로
    xgboost_file = "submissions/xgboost/xgboost_kfold_1019_1310.csv" # v6
    deepfm_file = "submissions/deepfm/deepfm_kfold_ensemble_20251019_125533.csv"  # v2.2 embedding

    # 데이터 로드
    xgb_df = pd.read_csv(xgboost_file)
    deepfm_df = pd.read_csv(deepfm_file)

    # ID 정렬 확인
    xgb_df = xgb_df.sort_values('ID').reset_index(drop=True)
    deepfm_df = deepfm_df.sort_values('ID').reset_index(drop=True)

    # 가중치 조합들
    weight_combinations = [
        (0.80, 0.20, "80_20"),
        (0.75, 0.25, "75_25"),
        (0.70, 0.30, "70_30"),
        (0.65, 0.35, "65_35"),
        (0.60, 0.40, "60_40"),
        (0.50, 0.50, "50_50"),
    ]

    timestamp = datetime.now().strftime('%m%d_%H%M')

    # 출력 디렉토리 생성
    output_dir = Path("submissions/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 조합별 앙상블 생성
    for xgb_weight, deepfm_weight, name in weight_combinations:
        ensemble_preds = (xgb_weight * xgb_df['clicked'] +
                         deepfm_weight * deepfm_df['clicked'])

        result_df = xgb_df[['ID']].copy()
        result_df['clicked'] = ensemble_preds

        filename = f"ensemble_xgb_deepfm_{name}_{timestamp}.csv"
        result_df.to_csv(f"submissions/ensemble/{filename}", index=False)

        print(f"{name}: Saved {filename}")

if __name__ == "__main__":
    main()