# Toss NEXT ML Challenge: 광고 CTR 예측

토스가 주최한 광고 클릭 예측 대회(2025, DACON 진행)에 개인으로 참가해 **상위 7%**에 들었습니다. XGBoost와 DeepFM을 각각 5-Fold로 학습하고 가중 평균한 앙상블이 최종 제출이며, 이 저장소의 `inference.py`는 최종 제출 파일을 그대로 다시 만들어 냅니다.

## 문제

사용자 속성, 광고 지면, 시간, 익명화된 수치 피처 약 120개로 광고 클릭 여부를 예측하는 이진 분류입니다.

| 구분 | 내용 |
| --- | --- |
| 범주형 피처 | gender, age_group, inventory_id, hour, day_of_week (5개) |
| 수치형 피처 | feat_a, feat_b, l_feat 계열 등 약 120개 |
| 추가 피처 | 지면 인기도와 빈도, 시간의 주기성 인코딩, 결측 여부 지시자 |

## 접근

| 모델 | 역할 | 설정 |
| --- | --- | --- |
| XGBoost | 수치 피처의 비선형 분기 | 5-Fold Stratified CV, lr 0.01, depth 7, subsample 0.8, GPU hist |
| DeepFM | 범주형 피처 간 2차 상호작용(FM)과 고차 패턴(MLP) | 동적 임베딩 `6 × cardinality^0.25` (4~50차원), MLP 256-128, dropout 0.25, AdamW + Cosine, early stopping 5 |
| 앙상블 | 두 모델의 서로 다른 오차 상쇄 | XGBoost 비중 50~80% 사이 6개 비율 비교 후 **65:35**로 최종 제출 |

DeepFM 입력은 QuantileTransformer로 스케일링했고, 두 모델 모두 Out-of-Fold 예측으로 검증했습니다.

## 결과

| 항목 | 값 |
| --- | --- |
| 최종 순위 | 상위 7% |
| DeepFM 5-Fold 대회 점수 | CV 0.3513 ± 0.0010, OOF 0.3512 |
| 재현 검증 | XGBoost, DeepFM 모두 원본 제출과 예측 차이 0, 앙상블 차이 약 1e-9 |

폴드 간 편차가 0.001 수준이라 검증이 안정적이라고 판단했습니다.

## 재현

```bash
conda create -n ctr python=3.10 && conda activate ctr
pip install -r requirements.txt
# GPU: pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python inference.py
```

`inference.py`는 시드, 전처리기, 체크포인트를 고정해 추론만 수행합니다. GPU에서 3~5분, CPU에서 30~40분 걸립니다. 학습 데이터와 체크포인트는 저장소에 포함하지 않았습니다. 학습 전체를 돌리려면 `src/deepfm/`, `src/xgboost/`의 prepare, train, predict 스크립트를 순서대로 실행한 뒤 `src/ensemble/two_model_ensemble.py`를 실행하면 됩니다.

## 구조

```
src/
├── deepfm/        전처리, K-Fold 학습, 예측
├── xgboost/       전처리, K-Fold 학습, 예측
├── models/        DeepFM 모델 정의, XGBoost 피처 프로세서
├── ensemble/      2-모델 가중 평균
└── utils/         지표, 설정
experiments/       Optuna 탐색, 피처 중요도 분석
inference.py       최종 제출 재현 스크립트
```

## 실서비스에서 다르게 할 것

대회에서는 오프라인 점수만 보면 되지만, 실제 광고나 추천 지면에서는 사용자 피로도와 노출 빈도 제약, 온라인 A/B 검증이 함께 필요합니다. 이 저장소는 모델링과 재현성에 집중한 버전입니다.
