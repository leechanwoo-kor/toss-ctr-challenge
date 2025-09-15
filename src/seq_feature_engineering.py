import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def create_sequence_features(df):
    """
    seq 컬럼에서 강력한 피처 추출

    발견사항 기반:
    - 긴 seq (>440): CTR 2.6%
    - 짧은 seq (≤440): CTR 1.2%
    - 평균 길이: ~508개 아이템
    """
    logger.info("Creating sequence features from seq column...")

    if 'seq' not in df.columns:
        logger.warning("seq column not found, skipping sequence features")
        return df

    # 결과를 저장할 dict (fragmentation 방지)
    new_features = {}

    # 메모리 효율적인 청크 기반 처리
    logger.info("Processing sequence data in memory-efficient chunks...")

    CHUNK_SIZE = 100000  # 10만 행씩 처리
    total_rows = len(df)

    def parse_seq_safe(seq_str):
        """안전한 seq 파싱"""
        if pd.isna(seq_str) or not isinstance(seq_str, str):
            return []
        try:
            return [int(x.strip()) for x in seq_str.split(',') if x.strip().isdigit()]
        except:
            return []

    # ================================
    # 청크별 피처 생성 및 통합
    # ================================
    logger.info("Creating features in chunks...")

    # 결과 저장용 리스트들
    seq_lengths = []
    seq_unique_counts = []
    seq_last_items = []
    seq_last_3_diversities = []
    seq_recent_repetitions = []
    seq_most_frequent_ratios = []
    seq_max_consecutives = []
    seq_diversity_changes = []

    # 청크별 처리
    for chunk_start in range(0, total_rows, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
        logger.info(f"Processing chunk {chunk_start//CHUNK_SIZE + 1}/{(total_rows-1)//CHUNK_SIZE + 1}")

        # 현재 청크의 seq 데이터 파싱
        chunk_seqs = df['seq'].iloc[chunk_start:chunk_end].apply(parse_seq_safe)

        # 기본 통계
        chunk_lengths = chunk_seqs.apply(len)
        chunk_unique_counts = chunk_seqs.apply(lambda x: len(set(x)) if x else 0)

        # 최근 행동 패턴
        chunk_last_items = chunk_seqs.apply(lambda x: x[-1] if x else -1)

        def last_3_diversity(seq):
            if len(seq) < 3:
                return 0
            last_3 = seq[-3:]
            return len(set(last_3)) / 3

        chunk_last_3_diversities = chunk_seqs.apply(last_3_diversity)

        def recent_repetition(seq, window=5):
            if len(seq) < window:
                return 0
            recent = seq[-window:]
            if not recent:
                return 0
            most_common_count = Counter(recent).most_common(1)[0][1]
            return most_common_count / window

        chunk_recent_repetitions = chunk_seqs.apply(recent_repetition)

        # 전체 패턴 분석
        def most_frequent_ratio(seq):
            if not seq:
                return 0
            counter = Counter(seq)
            most_common_count = counter.most_common(1)[0][1]
            return most_common_count / len(seq)

        chunk_most_frequent_ratios = chunk_seqs.apply(most_frequent_ratio)

        # 특별 패턴
        def max_consecutive(seq):
            if not seq:
                return 0
            max_count = 1
            current_count = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current_count += 1
                    max_count = max(max_count, current_count)
                else:
                    current_count = 1
            return max_count

        chunk_max_consecutives = chunk_seqs.apply(max_consecutive)

        # 행동 변화 패턴
        def diversity_change(seq):
            if len(seq) < 4:
                return 0
            mid = len(seq) // 2
            first_half = seq[:mid]
            second_half = seq[mid:]

            first_diversity = len(set(first_half)) / len(first_half) if first_half else 0
            second_diversity = len(set(second_half)) / len(second_half) if second_half else 0

            return second_diversity - first_diversity

        chunk_diversity_changes = chunk_seqs.apply(diversity_change)

        # 결과 저장
        seq_lengths.extend(chunk_lengths.values)
        seq_unique_counts.extend(chunk_unique_counts.values)
        seq_last_items.extend(chunk_last_items.values)
        seq_last_3_diversities.extend(chunk_last_3_diversities.values)
        seq_recent_repetitions.extend(chunk_recent_repetitions.values)
        seq_most_frequent_ratios.extend(chunk_most_frequent_ratios.values)
        seq_max_consecutives.extend(chunk_max_consecutives.values)
        seq_diversity_changes.extend(chunk_diversity_changes.values)

        # 메모리 정리
        del chunk_seqs, chunk_lengths, chunk_unique_counts
        del chunk_last_items, chunk_last_3_diversities, chunk_recent_repetitions
        del chunk_most_frequent_ratios, chunk_max_consecutives, chunk_diversity_changes

    # 결과를 numpy array로 변환하여 메모리 효율성 확보
    logger.info("Converting results to efficient data types...")

    new_features['seq_length'] = np.array(seq_lengths, dtype='int16')
    new_features['seq_unique_count'] = np.array(seq_unique_counts, dtype='int16')
    new_features['seq_last_item'] = np.array(seq_last_items, dtype='int16')
    new_features['seq_last_3_diversity'] = np.array(seq_last_3_diversities, dtype='float32')
    new_features['seq_recent_repetition'] = np.array(seq_recent_repetitions, dtype='float32')
    new_features['seq_most_frequent_ratio'] = np.array(seq_most_frequent_ratios, dtype='float32')
    new_features['seq_max_consecutive'] = np.array(seq_max_consecutives, dtype='int16')
    new_features['seq_diversity_change'] = np.array(seq_diversity_changes, dtype='float32')

    # 파생 피처들
    new_features['seq_diversity_ratio'] = np.where(
        new_features['seq_length'] > 0,
        new_features['seq_unique_count'] / new_features['seq_length'],
        0
    ).astype('float32')

    # 길이 기반 피처들
    median_length = np.median(new_features['seq_length'])
    length_90th = np.percentile(new_features['seq_length'], 90)
    length_10th = np.percentile(new_features['seq_length'], 10)

    new_features['seq_is_long'] = (new_features['seq_length'] > median_length).astype('uint8')
    new_features['seq_is_very_long'] = (new_features['seq_length'] > length_90th).astype('uint8')
    new_features['seq_is_very_short'] = (new_features['seq_length'] < length_10th).astype('uint8')

    # ================================
    # 추가 파생 피처들
    # ================================
    logger.info("Creating additional derived features...")

    # 단일 행동 세션 여부 (바운스)
    new_features['seq_is_single_action'] = (new_features['seq_length'] == 1).astype('uint8')

    # 탐색형 vs 집중형 행동 분류
    new_features['seq_is_exploration'] = (new_features['seq_diversity_ratio'] > 0.8).astype('uint8')
    new_features['seq_is_focused'] = (new_features['seq_diversity_ratio'] < 0.3).astype('uint8')

    # 연속 행동의 비율
    new_features['seq_consecutive_ratio'] = np.where(
        new_features['seq_length'] > 0,
        new_features['seq_max_consecutive'] / new_features['seq_length'],
        0
    ).astype('float32')

    # ================================
    # 비즈니스 관련 피처 (벡터화)
    # ================================
    logger.info("Creating business-relevant features...")

    # 세션 길이별 분류 (벡터화)
    lengths = new_features['seq_length']
    new_features['seq_session_type'] = np.where(
        lengths <= 5, 0,  # quick_browse
        np.where(lengths <= 50, 1,  # normal_session
                np.where(lengths <= 200, 2, 3))  # engaged_session or power_user
    ).astype('uint8')

    # 고유 아이템 수 기반 사용자 타입 (벡터화)
    diversity_ratios = new_features['seq_diversity_ratio']
    new_features['seq_user_type'] = np.where(
        new_features['seq_length'] == 0, 0,  # inactive
        np.where(diversity_ratios > 0.8, 3,  # explorer
                np.where(diversity_ratios > 0.5, 2,  # browser
                        np.where(diversity_ratios > 0.2, 1, 0)))  # focused or repetitive
    ).astype('uint8')

    # ================================
    # 결과 통합 및 반환
    # ================================
    logger.info("Integrating sequence features...")

    # 새 피처들을 DataFrame으로 만들어 한 번에 추가 (fragmentation 방지)
    new_df = pd.DataFrame(new_features, index=df.index)
    result_df = pd.concat([df, new_df], axis=1)

    # 로깅
    feature_count = len(new_features)
    logger.info(f"Created {feature_count} sequence features:")
    logger.info(f"  - Basic stats: seq_length, seq_unique_count, seq_diversity_ratio")
    logger.info(f"  - Recent behavior: seq_last_item, seq_recent_repetition")
    logger.info(f"  - Pattern analysis: seq_most_frequent_ratio, seq_consecutive_ratio")
    logger.info(f"  - Business types: seq_session_type, seq_user_type")

    # 결과 통계
    logger.info(f"Sequence feature statistics:")
    logger.info(f"  Long sequences (>{median_length:.0f}): {new_features['seq_is_long'].sum():,}")
    logger.info(f"  Very long sequences (>{length_90th:.0f}): {new_features['seq_is_very_long'].sum():,}")
    logger.info(f"  Exploration sessions: {new_features['seq_is_exploration'].sum():,}")
    logger.info(f"  Power users: {(new_features['seq_session_type'] == 3).sum():,}")

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