"""
评测模块：AUC + TPR @ 1% FPR。
返回 dict（除了 print 之外），便于 main.py 序列化到 metrics.json。
"""
import numpy as np
from sklearn.metrics import roc_curve, auc


class Evaluator:
    @staticmethod
    def calculate_metrics(y_true, y_scores) -> dict:
        """
        y_true: 真实标签，1=成员, 0=非成员
        y_scores: 攻击输出（似然比或 999）

        返回 {"auc": float, "tpr_at_1_fpr": float}
        """
        if len(set(y_true)) < 2:
            print(f"[Evaluator] 警告: y_true 单一类别，metrics 无效")
            return {"auc": float("nan"), "tpr_at_1_fpr": 0.0}

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        idx = np.where(fpr <= 0.01)[0]
        tpr_at_1_fpr = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

        print(f"[Evaluator] AUC = {roc_auc:.4f}  TPR@1%FPR = {tpr_at_1_fpr:.4f}")
        return {"auc": float(roc_auc), "tpr_at_1_fpr": tpr_at_1_fpr}
