"""
CTR 예측 평가 메트릭
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_competition_metrics(y_true, y_pred):
    """경쟁 지표 계산 (대회 채점 기준)"""
    # AP (Average Precision) - 50%
    ap = average_precision_score(y_true, y_pred)

    # WLL (Weighted LogLoss) - 50%
    eps = 1e-7  # 수치적 안정성을 위한 적절한 값
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred_clip[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred_clip[mask_1])) if mask_1.sum() > 0 else 0
    wll = 0.5 * ll_0 + 0.5 * ll_1

    # 대회 최종 점수 (AP 50% + WLL 50%)
    comp_score = 0.5 * ap + 0.5 * (1 / (1 + wll))

    # 추가 참고 메트릭
    auc = roc_auc_score(y_true, y_pred)
    logloss = -np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

    return {
        'comp_score': comp_score,  # 대회 최종 점수
        'ap': ap,                  # Average Precision (50%)
        'wll': wll,                # Weighted LogLoss (50%)
        'auc': auc,                # 참고용 AUC
        'logloss': logloss         # 참고용 LogLoss
    }