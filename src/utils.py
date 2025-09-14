import gc
import numpy as np
import psutil
from sklearn.metrics import average_precision_score

def print_memory(context=""):
    """현재 메모리 사용량을 출력하는 함수"""
    mem = psutil.virtual_memory()
    print(f"💾 {context} Memory: {mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")

def clear_memory():
    """메모리를 정리하는 함수"""
    gc.collect()
    print("🧹 Memory cleared")

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """대회 평가 산식인 Weighted LogLoss를 계산하는 함수"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0
    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """대회 최종 점수인 Competition Score를 계산하는 함수"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll
