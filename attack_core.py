"""
DC-MIA 攻击核心。

修复记录：
- Bug #1 (旧): phase_1_find_threshold 直接 return 0.9 —— 现已实现在 aux_data 上扫最优 τ₁
- Bug #2 (旧): 缺 τ₂ 判定 —— 现已新增 find_tau_2 + attack() 返回 (score, decision)
- Bug #3 (旧): phase_2 缺 per-sample 随机种子 —— 现已 attack() 首行 random.seed(sample_id)
"""
import random
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve
from rag_system import SimpleRAG


# 真实情况下应替换为 sentence_transformers.util.cos_sim
def calculate_similarity(pred: str, truth: str) -> float:
    return random.uniform(0.3, 0.95)


class DCMIA:
    """
    Difficulty-Calibrated MIA (DC-MIA) 攻击核心实现。
    复现自 RAG-leaks (Wang et al. 2025, Science China Information Sciences)。
    """

    def __init__(self, llm_service, data_pool, per_sample_seed: bool = True):
        self.llm_service = llm_service
        self.data_pool = data_pool
        self.per_sample_seed = per_sample_seed

    # ------------------------------------------------------------------ #
    # 阈值搜索
    # ------------------------------------------------------------------ #

    def phase_1_find_threshold(self, target_rag, aux_data, target_ids, tau_1_spec="auto") -> float:
        """
        阶段 1: 在辅助集上找最优高相似度阈值 τ₁（最大化 TPR @ 1% FPR）。
        aux_data 是已知 member/non-member 标签的样本。
        tau_1_spec: "auto" → 自动扫；或显式给 float 字符串
        """
        print(f"[DC-MIA] 阶段 1: 在 {len(aux_data)} 个 aux 样本上扫最优 τ₁...")

        if isinstance(tau_1_spec, str) and tau_1_spec != "auto":
            return float(tau_1_spec)

        sims, labels = [], []
        for s in aux_data:
            ans = target_rag.generate_answer(s["query"])
            sims.append(calculate_similarity(ans, s["answer"]))
            labels.append(1 if s["id"] in target_ids else 0)

        if not sims or len(set(labels)) < 2:
            print("[DC-MIA] 阶段 1: aux 太小或单类别，τ₁ 退到 0.9")
            return 0.9

        fpr, tpr, thr = roc_curve(labels, sims)
        mask = fpr <= 0.01
        if not mask.any():
            return float(thr[0])
        best_idx = mask.nonzero()[0][tpr[mask].argmax()]
        tau_1 = float(thr[best_idx])
        print(f"[DC-MIA] 阶段 1: 找到 τ₁ = {tau_1:.4f}")
        return tau_1

    def find_tau_2(self, target_rag, aux_data, target_ids, tau_1, m: int = 8, global_seed: int = 0) -> float:
        """
        在 aux_data 的"混淆区间"（actual_sim ≤ τ₁）上算似然比，
        跑 ROC 找最优 τ₂（按 TPR@1%FPR）。
        """
        print(f"[DC-MIA] 阶段 2: 在混淆区扫最优 τ₂ (τ₁={tau_1:.4f})...")

        # 1) 先算每个 aux 的 actual_sim，分出混淆区
        sims = []
        for s in aux_data:
            ans = target_rag.generate_answer(s["query"])
            sims.append(calculate_similarity(ans, s["answer"]))
        confusion_mask = np.array(sims) <= tau_1
        aux_in_confusion = [s for s, m_ in zip(aux_data, confusion_mask) if m_]
        print(f"[DC-MIA] 阶段 2: aux 落在混淆区 {len(aux_in_confusion)}/{len(aux_data)}")

        if not aux_in_confusion:
            print("[DC-MIA] 阶段 2: 无混淆样本，τ₂ 退到 1.0")
            return 1.0

        # 2) 对混淆区每个 aux 算似然比
        lrs, labels = [], []
        for s in aux_in_confusion:
            if self.per_sample_seed:
                random.seed(global_seed * 1_000_003 + s["id"])
            lr = self._compute_lr(target_rag, s, m=m)
            lrs.append(lr)
            labels.append(1 if s["id"] in target_ids else 0)

        if len(set(labels)) < 2:
            print("[DC-MIA] 阶段 2: 混淆区单类别，τ₂ 退到 1.0")
            return 1.0

        fpr, tpr, thr = roc_curve(labels, lrs)
        mask = fpr <= 0.01
        if not mask.any():
            return float(thr[0])
        best_idx = mask.nonzero()[0][tpr[mask].argmax()]
        tau_2 = float(thr[best_idx])
        print(f"[DC-MIA] 阶段 2: 找到 τ₂ = {tau_2:.4f}")
        return tau_2

    # ------------------------------------------------------------------ #
    # 似然比计算（被 attack() 和 find_tau_2() 共用）
    # ------------------------------------------------------------------ #

    def _compute_lr(self, target_rag, target_sample, m: int) -> float:
        """纯算似然比（无 τ₁/τ₂ 判定），给阶段 2 和 find_tau_2 共用。"""
        ans_target = target_rag.generate_answer(target_sample["query"])
        actual_sim = calculate_similarity(ans_target, target_sample["answer"])

        in_sims, out_sims = [], []
        for _ in range(m):
            bg = random.sample(self.data_pool, 8000) if len(self.data_pool) >= 8000 else list(self.data_pool)

            out_rag = SimpleRAG(self.llm_service)
            out_rag.build_index(bg)
            out_sims.append(calculate_similarity(
                out_rag.generate_answer(target_sample["query"]), target_sample["answer"]))

            in_data = bg.copy()
            in_data[0] = target_sample
            in_rag = SimpleRAG(self.llm_service)
            in_rag.build_index(in_data)
            in_sims.append(calculate_similarity(
                in_rag.generate_answer(target_sample["query"]), target_sample["answer"]))

        mu_in, std_in = norm.fit(in_sims)
        mu_out, std_out = norm.fit(out_sims)
        return norm.pdf(actual_sim, mu_in, std_in + 1e-9) / (norm.pdf(actual_sim, mu_out, std_out + 1e-9) + 1e-9)

    # ------------------------------------------------------------------ #
    # 单样本攻击
    # ------------------------------------------------------------------ #

    def attack(self, target_rag, target_sample, tau_1: float, tau_2: float, m: int = 8, global_seed: int = 0):
        """
        执行完整的两阶段攻击，返回 (score, decision)：
          score: 999.0 表示阶段 1 命中（高相似度），否则是阶段 2 的似然比
          decision: 1=成员, 0=非成员
        """
        if self.per_sample_seed:
            random.seed(global_seed * 1_000_003 + target_sample["id"])

        ans = target_rag.generate_answer(target_sample["query"])
        actual_sim = calculate_similarity(ans, target_sample["answer"])

        if actual_sim > tau_1:
            return 999.0, 1   # 阶段 1: 高相似度直接判成员

        lr = self._compute_lr(target_rag, target_sample, m=m)
        return lr, int(lr > tau_2)
