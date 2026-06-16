"""
数据管理模块：负责从本地/dummy 加载数据，按 config 切分数据集。

约定切分（与论文 RAG-leaks §5.1 一致）：
  - target_kb: target_kb_size（论文 8000）
  - eval:      eval_members + eval_non_members（论文 1000+1000）
  - aux:       aux_members + aux_non_members（论文 100+100）
  - reference_pool: 剩余非成员，给 phase_2 影子 RAG 采样
"""
import os
import json
import random


class DataManager:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"[DataManager] 创建了空的数据目录: {self.data_dir}")

    def load_or_generate_dummy_data(
        self,
        total_samples: int = 12000,
        dataset_name: str = "dummy",
        hf_dataset_id=None,
    ):
        """
        加载数据。当前实现只支持 dummy（生成假数据）；
        真实数据集（healthcaremagic/agnews/nq）的 HF 接入留给后续 PR。
        """
        data_file = os.path.join(self.data_dir, "dataset.json")
        if os.path.exists(data_file):
            print(f"[DataManager] 从 {data_file} 加载数据...")
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)

        if dataset_name != "dummy":
            print(f"[DataManager] 真实数据集 '{dataset_name}' 加载尚未实现，"
                  f"回退到 dummy (total_samples={total_samples})")
        else:
            print(f"[DataManager] 生成 {total_samples} 条 dummy 数据...")

        dummy_data = []
        for i in range(total_samples):
            dummy_data.append({
                "id": i,
                "query": f"Dummy query for sample {i}",
                "answer": f"Dummy ground truth answer for sample {i}",
            })
        return dummy_data

    def split_data(
        self,
        data: list,
        target_kb_size: int = 8000,
        eval_members: int = 1000,
        eval_non_members: int = 1000,
        aux_members: int = 100,
        aux_non_members: int = 100,
    ):
        """
        按论文比例切分。最少需要 target_kb_size + eval_members + eval_non_members + aux_members + aux_non_members + 一个最小 reference_pool。
        """
        min_required = target_kb_size + eval_members + eval_non_members + aux_members + aux_non_members + 100
        if len(data) < min_required:
            raise ValueError(
                f"[DataManager] 数据量不足: 当前 {len(data)} 条, 最少需要 {min_required} 条。"
            )

        random.seed(42)
        random.shuffle(data)

        members = data[:target_kb_size]
        non_members = data[target_kb_size:]

        eval_data = members[:eval_members] + non_members[:eval_non_members]
        aux_data = members[eval_members:eval_members + aux_members] \
                 + non_members[eval_non_members:eval_non_members + aux_non_members]
        reference_pool = non_members[eval_non_members + aux_non_members:]

        return {
            "target_kb": members,
            "eval_data": eval_data,
            "aux_data": aux_data,
            "reference_pool": reference_pool,
        }
