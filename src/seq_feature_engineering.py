import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 50000  # 메모리 사용량 줄이기
LONG_SEQ_THRESHOLD = 440
DIVERSITY_HIGH_THRESHOLD = 0.8
DIVERSITY_LOW_THRESHOLD = 0.3
RECENT_WINDOW = 5

def parse_seq_safe(seq_str):
    """안전한 seq 파싱"""
    if pd.isna(seq_str) or not isinstance(seq_str, str):
        return []
    try:
        return [int(x.strip()) for x in seq_str.split(',') if x.strip().isdigit()]
    except:
        return []

def last_3_diversity(seq):
    if len(seq) < 3:
        return 0
    return len(set(seq[-3:])) / 3

def recent_repetition(seq, window=RECENT_WINDOW):
    if len(seq) < window:
        return 0
    recent = seq[-window:]
    return Counter(recent).most_common(1)[0][1] / window if recent else 0

def most_frequent_ratio(seq):
    if not seq:
        return 0
    return Counter(seq).most_common(1)[0][1] / len(seq)

def max_consecutive(seq):
    if not seq:
        return 0
    max_count = current_count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 1
    return max_count

def diversity_change(seq):
    if len(seq) < 4:
        return 0
    mid = len(seq) // 2
    first_half, second_half = seq[:mid], seq[mid:]
    first_div = len(set(first_half)) / len(first_half) if first_half else 0
    second_div = len(set(second_half)) / len(second_half) if second_half else 0
    return second_div - first_div

def recent_5_diversity(seq):
    """최근 5개 행동의 다양성"""
    if len(seq) < 5:
        return len(set(seq)) / len(seq) if seq else 0
    recent_5 = seq[-5:]
    return len(set(recent_5)) / 5

def process_chunk_features(chunk_seqs):
    """청크별 피처 계산"""
    return {
        'lengths': chunk_seqs.apply(len),
        'unique_counts': chunk_seqs.apply(lambda x: len(set(x)) if x else 0),
        'last_items': chunk_seqs.apply(lambda x: x[-1] if x else -1),
        'last_3_diversities': chunk_seqs.apply(last_3_diversity),
        'recent_repetitions': chunk_seqs.apply(recent_repetition),
        'most_frequent_ratios': chunk_seqs.apply(most_frequent_ratio),
        'max_consecutives': chunk_seqs.apply(max_consecutive),
        'diversity_changes': chunk_seqs.apply(diversity_change),
        'recent_5_diversities': chunk_seqs.apply(recent_5_diversity)  # 🔥 NEW
    }

def create_derived_features(base_features):
    """기본 피처에서 파생 피처 생성"""
    features = base_features.copy()

    # 다양성 비율
    features['seq_diversity_ratio'] = np.where(
        features['seq_length'] > 0,
        features['seq_unique_count'] / features['seq_length'], 0
    ).astype('float32')

    # 길이 기반 분류
    median_len = np.median(features['seq_length'])
    p90, p10 = np.percentile(features['seq_length'], [90, 10])
    features['seq_is_long'] = (features['seq_length'] > median_len).astype('uint8')
    features['seq_is_very_long'] = (features['seq_length'] > p90).astype('uint8')
    features['seq_is_very_short'] = (features['seq_length'] < p10).astype('uint8')

    # 행동 패턴 분류
    features['seq_is_single_action'] = (features['seq_length'] == 1).astype('uint8')
    features['seq_is_exploration'] = (features['seq_diversity_ratio'] > DIVERSITY_HIGH_THRESHOLD).astype('uint8')
    features['seq_is_focused'] = (features['seq_diversity_ratio'] < DIVERSITY_LOW_THRESHOLD).astype('uint8')

    # 연속성 비율
    features['seq_consecutive_ratio'] = np.where(
        features['seq_length'] > 0,
        features['seq_max_consecutive'] / features['seq_length'], 0
    ).astype('float32')

    # 세션 타입 (0: quick, 1: normal, 2: engaged, 3: power)
    lengths = features['seq_length']
    features['seq_session_type'] = np.where(
        lengths <= 5, 0, np.where(lengths <= 50, 1, np.where(lengths <= 200, 2, 3))
    ).astype('uint8')

    # 사용자 타입 (0: inactive/repetitive, 1: focused, 2: browser, 3: explorer)
    diversity = features['seq_diversity_ratio']
    features['seq_user_type'] = np.where(
        features['seq_length'] == 0, 0,
        np.where(diversity > 0.8, 3, np.where(diversity > 0.5, 2, np.where(diversity > 0.2, 1, 0)))
    ).astype('uint8')

    # 🔥 NEW: 강력한 CTR 예측 피처들
    # Ultra-short 시퀀스 (CTR 0.043 vs 0.019 = 2.2배)
    features['seq_is_ultra_short'] = (features['seq_length'] <= 10).astype('uint8')

    # High diversity with reasonable threshold (CTR 1.7배 차이)
    features['seq_high_diversity'] = (features['seq_diversity_ratio'] > 0.3).astype('uint8')

    # Recent behavior focus (최근 5개 행동 다양성)
    # Note: 이 피처는 실제 seq parsing이 필요하므로 create_sequence_features에서 구현

    return features

def create_sequence_features(df):
    """
    seq 컬럼에서 피처 추출
    긴 seq (>440): CTR 2.6%, 짧은 seq (≤440): CTR 1.2%
    """
    logger.info("Creating sequence features from seq column...")

    if 'seq' not in df.columns:
        logger.warning("seq column not found, skipping sequence features")
        return df

    total_rows = len(df)
    feature_lists = {name: [] for name in [
        'lengths', 'unique_counts', 'last_items', 'last_3_diversities',
        'recent_repetitions', 'most_frequent_ratios', 'max_consecutives', 'diversity_changes',
        'recent_5_diversities'  # 🔥 NEW
    ]}

    # 청크별 처리
    for chunk_start in range(0, total_rows, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
        logger.info(f"Processing chunk {chunk_start//CHUNK_SIZE + 1}/{(total_rows-1)//CHUNK_SIZE + 1}")

        chunk_seqs = df['seq'].iloc[chunk_start:chunk_end].apply(parse_seq_safe)
        chunk_features = process_chunk_features(chunk_seqs)

        for name, values in chunk_features.items():
            feature_lists[name].extend(values.values)

    # 기본 피처 생성
    base_features = {
        'seq_length': np.array(feature_lists['lengths'], dtype='int16'),
        'seq_unique_count': np.array(feature_lists['unique_counts'], dtype='int16'),
        'seq_last_item': np.array(feature_lists['last_items'], dtype='int16'),
        'seq_last_3_diversity': np.array(feature_lists['last_3_diversities'], dtype='float32'),
        'seq_recent_repetition': np.array(feature_lists['recent_repetitions'], dtype='float32'),
        'seq_most_frequent_ratio': np.array(feature_lists['most_frequent_ratios'], dtype='float32'),
        'seq_max_consecutive': np.array(feature_lists['max_consecutives'], dtype='int16'),
        'seq_diversity_change': np.array(feature_lists['diversity_changes'], dtype='float32'),
        'seq_recent_5_diversity': np.array(feature_lists['recent_5_diversities'], dtype='float32')  # 🔥 NEW
    }

    # 파생 피처 추가
    all_features = create_derived_features(base_features)

    # 메모리 효율적으로 피처 추가
    logger.info("Adding sequence features to original DataFrame...")
    for feature_name, feature_values in all_features.items():
        df[feature_name] = feature_values
        # 메모리 절약을 위해 적절한 dtype 확인
        if df[feature_name].dtype == 'int64':
            if df[feature_name].min() >= 0 and df[feature_name].max() <= 255:
                df[feature_name] = df[feature_name].astype('uint8')
            elif df[feature_name].min() >= -128 and df[feature_name].max() <= 127:
                df[feature_name] = df[feature_name].astype('int8')
            elif df[feature_name].min() >= -32768 and df[feature_name].max() <= 32767:
                df[feature_name] = df[feature_name].astype('int16')

    result_df = df

    logger.info(f"Created {len(all_features)} sequence features")
    return result_df

def quick_sequence_analysis(df, sample_size=5000):
    """
    seq 피처들의 CTR 예측력 빠른 검증
    """
    logger.info(f"Quick sequence feature validation with {sample_size} samples...")

    if 'clicked' not in df.columns:
        logger.warning("No 'clicked' column found for validation")
        return

    # 샘플링
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)

    # seq 피처들만 선택
    seq_features = [col for col in sample_df.columns if col.startswith('seq_')]

    if not seq_features:
        logger.warning("No sequence features found")
        return

    logger.info(f"Analyzing {len(seq_features)} sequence features...")

    # 각 피처별 CTR 차이 분석
    for feature in seq_features:
        try:
            if sample_df[feature].dtype in ['uint8', 'int8']:
                # 바이너리나 카테고리 피처
                ctr_by_value = sample_df.groupby(feature)['clicked'].agg(['count', 'mean'])
                if len(ctr_by_value) <= 5:  # 카테고리가 적으면 출력
                    logger.info(f"  {feature}:")
                    for idx, row in ctr_by_value.iterrows():
                        logger.info(f"    {idx}: {row['mean']:.4f} CTR ({row['count']} samples)")
            else:
                # 연속형 피처 - 상위/하위 비교
                median_val = sample_df[feature].median()
                high_ctr = sample_df[sample_df[feature] > median_val]['clicked'].mean()
                low_ctr = sample_df[sample_df[feature] <= median_val]['clicked'].mean()
                ratio = high_ctr / low_ctr if low_ctr > 0 else 0
                logger.info(f"  {feature}: High({high_ctr:.4f}) vs Low({low_ctr:.4f}) = {ratio:.2f}x")

        except Exception as e:
            logger.warning(f"  {feature}: Analysis failed - {e}")

if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)

    print("Testing sequence feature engineering...")

    # 샘플 데이터로 테스트
    df = pd.read_parquet('data/train.parquet').sample(1000, random_state=42)

    # seq 피처 생성
    result_df = create_sequence_features(df)

    # 빠른 검증
    quick_sequence_analysis(result_df)

    print(f"\nResult shape: {result_df.shape}")
    print("Sequence features created successfully!")