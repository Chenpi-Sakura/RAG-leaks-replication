"""
S-MIA 基线（Li et al. 2024）。

论文 §5.3 定义：文本分段输入生成，以生成文本与后半段标准答案的余弦相似度作为成员分数。

算法（论文 §5.3）：
    对每个目标样本 x = (q, a):
      1. 取 a 的前半段作为 query'，a 的后半段作为 truth'
      2. 用 query' 调 target_rag.generate_answer → 得到 a'
      3. score = cosine_sim(a', truth')
      4. 阈值从 aux 集扫（按 1%FPR-TPR 最优）

注：复用 `BaseRetriever.encode` 算 cosine；与 DC-MIA 共享 retriever 实例。
"""
import numpy as np
from sklearn.metrics import roc_curve


class SMIA:
    """S-MIA 基线（Li 2024）：文本分段 → RAG 生成 → cosine sim。"""

    def __init__(self, retriever):
        self.retriever = retriever

    def cosine(self, a: str, b: str) -> float:
        a_emb = self.retriever.encode(a)
        b_emb = self.retriever.encode(b)
        return float(np.dot(a_emb, b_emb))

    @staticmethod
    def _split_text(text: str, mode: str = "char") -> tuple:
        """
        把文本分成前后两半。
        mode='char': 按字符数对半（适合 HealthCareMagic 等长度均匀的）
        mode='word': 按词数对半（适合 AgNews 等按词组织的）
        """
        if mode == "word":
            words = text.split()
            if len(words) < 4:
                return text, ""
            mid = len(words) // 2
            return " ".join(words[:mid]), " ".join(words[mid:])
        # 默认按字符
        if len(text) < 4:
            return text, ""
        mid = len(text) // 2
        return text[:mid], text[mid:]

    def find_threshold(self, target_rag, aux_data, target_ids, split_mode="char"):
        """在 aux_data 上扫描最优 τ_s（按 1%FPR-TPR）。"""
        sims, labels = [], []
        for s in aux_data:
            text = s["answer"]
            q1, t1 = self._split_text(text, mode=split_mode)
            if not q1 or not t1:
                continue
            generated = target_rag.generate_answer(q1)
            sims.append(self.cosine(generated, t1))
            labels.append(1 if s["id"] in target_ids else 0)
        if not sims or len(set(labels)) < 2:
            print("[S-MIA] aux 太小或单类别，τ_s 退到 0.5")
            return 0.5
        fpr, tpr, thr = roc_curve(labels, sims)
        mask = fpr <= 0.01
        if not mask.any():
            return float(thr[0])
        best_idx = mask.nonzero()[0][tpr[mask].argmax()]
        tau_s = float(thr[best_idx])
        print(f"[S-MIA] 找到 τ_s = {tau_s:.4f}")
        return tau_s

    def attack(self, target_rag, sample, split_mode="char"):
        """对单样本攻击，返回 (score, decision)。"""
        text = sample["answer"]
        q1, t1 = self._split_text(text, mode=split_mode)
        if not q1 or not t1:
            return 0.0, 0
        generated = target_rag.generate_answer(q1)
        score = self.cosine(generated, t1)
        return score, 0  # decision 由 main.py 拿 τ_s 后判定
