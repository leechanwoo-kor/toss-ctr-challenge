"""
Compare original submission with reproduced predictions
"""
import pandas as pd
import numpy as np

# 원본 제출 파일
original = pd.read_csv('submissions/ensemble_xgb_deepfm_65_35_1012_1728.csv')
# 재생성 파일
reproduced = pd.read_csv('submissions/ensemble/ensemble_xgb_deepfm_65_35_1017_2156.csv')

print('='*70)
print('재현성 검증 결과')
print('='*70)
print(f'원본 파일 크기: {len(original):,} rows')
print(f'재생성 파일 크기: {len(reproduced):,} rows')
print()

# ID 정렬
original = original.sort_values('ID').reset_index(drop=True)
reproduced = reproduced.sort_values('ID').reset_index(drop=True)

# 예측값 차이 계산
diff = np.abs(original['clicked'] - reproduced['clicked'])

print('예측값 차이 통계:')
print(f'  Max difference: {diff.max():.8f}')
print(f'  Mean difference: {diff.mean():.8f}')
print(f'  Median difference: {diff.median():.8f}')
print(f'  Std difference: {diff.std():.8f}')
print()

# 통계 비교
print('원본 파일 통계:')
print(f'  Min: {original["clicked"].min():.6f}')
print(f'  Max: {original["clicked"].max():.6f}')
print(f'  Mean: {original["clicked"].mean():.6f}')
print(f'  Std: {original["clicked"].std():.6f}')
print()

print('재생성 파일 통계:')
print(f'  Min: {reproduced["clicked"].min():.6f}')
print(f'  Max: {reproduced["clicked"].max():.6f}')
print(f'  Mean: {reproduced["clicked"].mean():.6f}')
print(f'  Std: {reproduced["clicked"].std():.6f}')
print()

# 결론
if diff.max() < 0.01:
    print('[OK] Reproduction successful! Difference < 1%')
elif diff.max() < 0.05:
    print('[WARNING] Partial reproduction: Difference < 5%')
else:
    print('[FAIL] Reproduction failed: Difference >= 5%')
