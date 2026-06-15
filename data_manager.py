import os
import json
import random

class DataManager:
    """
    数据管理模块，负责从本地读取数据，并按照论文设定划分数据集。
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"[DataManager] 创建了空的数据目录: {self.data_dir}")

    def load_or_generate_dummy_data(self, total_samples: int = 12000):
        """
        从本地加载数据。如果本地没数据，为了测试跑通，会生成一组 Dummy 数据。
        每条数据结构: {"id": int, "query": str, "answer": str}
        """
        data_file = os.path.join(self.data_dir, "dataset.json")
        if os.path.exists(data_file):
            print(f"[DataManager] 正在从 {data_file} 加载数据...")
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"[DataManager] 未找到本地数据，生成 {total_samples} 条 Dummy 数据供测试...")
            dummy_data = []
            for i in range(total_samples):
                dummy_data.append({
                    "id": i,
                    "query": f"Dummy query for sample {i}",
                    "answer": f"Dummy ground truth answer for sample {i}"
                })
            return dummy_data

    def split_data(self, data: list):
        """
        按照论文中的比例划分数据集：
        - 目标知识库 (target_kb): 8000
        - 辅助集 (auxiliary): 100 成员 + 100 非成员
        - 测试集 (eval): 1000 成员 + 1000 非成员
        - 剩余非成员用于构建 outRAG 的采样池 (reference_pool)
        """
        # 修复 Bug P2: 切分逻辑要求 total >= 8000+1000+200+1100 = 10300
        # 不足时直接抛清晰错误，避免 reference_pool 为空 phase 2 崩在 random.sample
        MIN_REQUIRED = 10300
        if len(data) < MIN_REQUIRED:
            raise ValueError(
                f"[DataManager] 数据量不足: 当前 {len(data)} 条, "
                f"最少需要 {MIN_REQUIRED} 条 (切分比例: 8000+1000+200+1100)。"
                f"请在 main.py 把 total_samples 调到 >= {MIN_REQUIRED}。"
            )

        # 确保随机性可复现
        random.seed(42)
        random.shuffle(data)
        
        # 前 8000 作为 member (属于 Target RAG)
        members = data[:8000]
        non_members = data[8000:]
        
        # 划分评估集 (1000 member, 1000 non-member)
        eval_members = members[:1000]
        eval_non_members = non_members[:1000]
        eval_data = eval_members + eval_non_members
        
        # 划分辅助集 (100 member, 100 non-member)
        aux_members = members[1000:1100]
        aux_non_members = non_members[1000:1100]
        aux_data = aux_members + aux_non_members
        
        # 构建影子知识库所需的数据池 (不含当前的 target test samples)
        # 为了简单，此处直接返回所有非成员，供后续采样
        reference_pool = non_members[1100:]
        
        return {
            "target_kb": members,
            "eval_data": eval_data,
            "aux_data": aux_data,
            "reference_pool": reference_pool
        }
