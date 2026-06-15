import numpy as np
from sklearn.metrics import roc_curve, auc

class Evaluator:
    """
    负责评测指标计算 (AUC, TPR @ 1% FPR)
    """
    @staticmethod
    def calculate_metrics(y_true, y_scores):
        """
        y_true: 真实标签，1 为成员，0 为非成员
        y_scores: 攻击输出的置信度/分数 (例如似然比)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 寻找 FPR <= 0.01 时的最大 TPR
        # tpr_at_1_fpr
        idx = np.where(fpr <= 0.01)[0]
        if len(idx) > 0:
            tpr_at_1_fpr = tpr[idx[-1]]
        else:
            tpr_at_1_fpr = 0.0
            
        print(f"[Evaluator] 评测完成: AUC = {roc_auc:.4f}, TPR @ 1% FPR = {tpr_at_1_fpr:.4f}")
        return roc_auc, tpr_at_1_fpr
