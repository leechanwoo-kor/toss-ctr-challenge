"""
CTR 예측 평가 메트릭
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_competition_metrics(y_true, y_pred):
    """경쟁 지표 계산"""
    ap = average_precision_score(y_true, y_pred)

    eps = 1e-15
    y_pred_clip = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred_clip[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred_clip[mask_1])) if mask_1.sum() > 0 else 0
    wll = 0.5 * ll_0 + 0.5 * ll_1

    comp_score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    auc = roc_auc_score(y_true, y_pred)

    return comp_score, ap, wll, auc