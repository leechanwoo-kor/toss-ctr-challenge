"""
시퀀스 피처 처리 유틸리티
"""

import numpy as np
import pandas as pd


SEQUENCE_FEATURES = [
    "seq_length",
    "seq_mean",
    "seq_std",
    "seq_min",
    "seq_max",
    "seq_normalized_range",
    "seq_cv",
]


def process_sequence_features(seq_series, batch_size=500000):
    """시퀀스를 통계 피처로 변환

    Args:
        seq_series: pandas Series containing sequence strings
        batch_size: 배치 처리 크기

    Returns:
        DataFrame with sequence statistical features
    """
    print("Processing sequence features...")

    def extract_stats(seq_str):
        """단일 시퀀스에서 통계 피처 추출"""
        if pd.isna(seq_str) or seq_str == "" or seq_str == "nan":
            return [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            seq_array = np.fromstring(str(seq_str), sep=",", dtype=np.float32)
            if len(seq_array) == 0:
                return [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            length = len(seq_array)
            mean = float(np.mean(seq_array))
            std = float(np.std(seq_array))
            minimum = float(np.min(seq_array))
            maximum = float(np.max(seq_array))

            # 최적화된 통계 피처 (중복 제거)
            range_val = maximum - minimum if maximum != minimum else 0
            normalized_range = range_val / (maximum + 1e-8) if maximum > 0 else 0
            cv = std / (mean + 1e-8) if mean != 0 else 0

            return [length, mean, std, minimum, maximum, normalized_range, cv]

        except Exception:
            return [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 배치 처리로 메모리 효율성 향상
    all_stats = []

    for i in range(0, len(seq_series), batch_size):
        batch_end = min(i + batch_size, len(seq_series))
        batch_seq = seq_series.iloc[i:batch_end]
        batch_stats = [extract_stats(seq) for seq in batch_seq]
        all_stats.extend(batch_stats)

        if i % batch_size == 0:
            print(f"  Processed {batch_end:,}/{len(seq_series):,} sequences ({batch_end/len(seq_series)*100:.1f}%)")

    # DataFrame 생성
    seq_features = pd.DataFrame(all_stats, columns=SEQUENCE_FEATURES, dtype=np.float32)

    print(f"  Created {seq_features.shape[1]} sequence features")
    return seq_features