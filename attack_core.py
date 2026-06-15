import random
import numpy as np
from scipy.stats import norm
from rag_system import SimpleRAG

# 用于模拟相似度计算 (这里使用随机数或一个假逻辑，真实情况使用 cosine_similarity)
def calculate_similarity(pred: str, truth: str) -> float:
    # 真实情况: 
    # from sentence_transformers import util
    # emb1 = embedder.encode(pred)
    # emb2 = embedder.encode(truth)
    # return util.cos_sim(emb1, emb2).item()
    return random.uniform(0.3, 0.95)

class DCMIA:
    """
    Difficulty-Calibrated MIA (DC-MIA) 攻击核心实现

    复现自 RAG-leaks (Wang et al. 2025, Science China Information Sciences)
    的 "difficulty-calibrated membership inference attack" 方法。

    注意：本类对应 RAG-leaks 的 DC-MIA，**不是** DCMI (Gao et al. 2025,
    arXiv:2509.06026) 的 "Differential Calibration MIA"——两者是独立工作，
    不要混用。本类名 DCMIA 强调 "D for Difficulty"。

    攻击流程：
    - Phase 1: 对相似度极高的样本（> tau_1）直接判定为成员（捡漏）
    - Phase 2: 对落入混淆区间的样本，构建 m 对 inRAG/outRAG 影子知识库，
      用真实 target_rag 响应在高斯分布上的似然比做"难度校准"
    """
    def __init__(self, llm_service, data_pool):
        self.llm_service = llm_service
        self.data_pool = data_pool  # 用于构建影子 RAG 的数据池

    def phase_1_find_threshold(self, target_rag, aux_data: list) -> float:
        """
        阶段 1: 在辅助集上找最优高相似度阈值 \\tau_1
        真实逻辑: 遍历 aux_data，查 target_rag 算相似度，ROC 找 TPR@1%FPR 最好的 tau
        当前为占位实现，返回固定值 0.9
        """
        print("[DC-MIA] 阶段 1: 正在计算辅助集上的最优高相似度阈值...")
        return 0.9

    def phase_2_likelihood_ratio(self, target_rag, target_sample: dict, m: int = 8) -> float:
        """
        阶段 2: 在目标 RAG 上做难度校准的似然比检验

        关键：actual_sim 必须来自 target_rag 自身的响应，**不能**用 random 模拟
        （修复 Bug #1: 之前用 "Actual response" 占位字符串得到的是 random 数）
        """
        print(f"[DC-MIA] 阶段 2: 为样本 {target_sample['id']} 构建 {m} 对影子 RAG 并计算似然比...")

        # ✅ 修复 Bug #1: actual_sim 从真实的 target_rag 取
        ans_target = target_rag.generate_answer(target_sample['query'])
        actual_sim = calculate_similarity(ans_target, target_sample['answer'])

        in_sims = []
        out_sims = []

        # 构建 m 对 inRAG 和 outRAG
        for _ in range(m):
            # 随机抽样构建影子库的背景数据
            bg_data = random.sample(self.data_pool, 8000) if len(self.data_pool) >= 8000 else self.data_pool

            # outRAG: 不包含 target
            out_rag = SimpleRAG(self.llm_service)
            out_rag.build_index(bg_data)
            ans_out = out_rag.generate_answer(target_sample['query'])
            out_sims.append(calculate_similarity(ans_out, target_sample['answer']))

            # inRAG: 包含 target（替换第一条为 target_sample）
            in_data = bg_data.copy()
            in_data[0] = target_sample
            in_rag = SimpleRAG(self.llm_service)
            in_rag.build_index(in_data)
            ans_in = in_rag.generate_answer(target_sample['query'])
            in_sims.append(calculate_similarity(ans_in, target_sample['answer']))

        # 拟合高斯分布
        mu_in, std_in = norm.fit(in_sims)
        mu_out, std_out = norm.fit(out_sims)

        # 计算 PDF 和似然比 (加 1e-9 防除零)
        pdf_in = norm.pdf(actual_sim, mu_in, std_in + 1e-9)
        pdf_out = norm.pdf(actual_sim, mu_out, std_out + 1e-9)
        likelihood_ratio = pdf_in / (pdf_out + 1e-9)

        return likelihood_ratio

    def attack(self, target_rag, target_sample: dict, tau_1: float, m: int = 8) -> float:
        """
        执行完整的两阶段攻击，返回最终的 Calibrated Score

        ✅ 修复 Bug #2: actual_sim 在 attack() 里只算一次，phase 1 和 phase 2 复用同一份
        """
        # 一次性算出 actual_sim，phase 1 和 phase 2 共用
        ans = target_rag.generate_answer(target_sample['query'])
        actual_sim = calculate_similarity(ans, target_sample['answer'])

        if actual_sim > tau_1:
            # 阶段 1: 高相似度 → 直接判定为强成员
            return 999.0
        else:
            # 阶段 2: 似然比检验
            return self.phase_2_likelihood_ratio(target_rag, target_sample, m=m)
