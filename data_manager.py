"""
数据管理模块：负责从本地/HF/dummy 加载数据，按 config 切分数据集。

约定切分（与论文 RAG-leaks §5.1 一致）：
  - target_kb: target_kb_size（论文 8000）
  - eval:      eval_members + eval_non_members（论文 1000+1000）
  - aux:       aux_members + aux_non_members（论文 100+100）
  - reference_pool: 剩余非成员，给 phase_2 影子 RAG 采样

支持的 dataset_name：
  dummy            — 假数据
  healthcaremagic  — 医疗对话（论文 §5.1.1）
  agnews           — 新闻文本（论文 §5.1.2：前 10 词 = query，剩余 = answer）
  naturalquestions — 通用 QA（论文 §5.1.3 / §7.3）

镜像支持：见 README "数据集加载" 章节；推荐用 hf-mirror.com 下载后存到 ./data/ 走 local 模式。
"""
import os
import json
import random


# 留个 hook：如果用户在 main.py 入口前 patch 了 HF_ENDPOINT（但 datasets 缓存问题
# 已证实无法只靠 env var 解决），这里尝试用 MonkeyPatch 注入：
def _try_setup_hf_mirror():
    endpoint = os.environ.get("HF_ENDPOINT")
    if not endpoint:
        return
    endpoint = endpoint.rstrip("/")
    try:
        import huggingface_hub.constants as _c
        _c.ENDPOINT = endpoint
        _c.HUGGINGFACE_CO_URL_TEMPLATE = endpoint + "/{repo_id}/resolve/{revision}/{filename}"
    except Exception:
        pass
    try:
        import datasets.config as _dc
        _dc.HF_ENDPOINT = endpoint
    except Exception:
        pass
    print(f"[DataManager] HF endpoint attempt -> {endpoint}")


_try_setup_hf_mirror()


class DataManager:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"[DataManager] 创建了空的数据目录: {self.data_dir}")

    # ------------------------------------------------------------------ #
    # 入口
    # ------------------------------------------------------------------ #

    def load_or_generate_dummy_data(
        self,
        total_samples: int = 12000,
        dataset_name: str = "dummy",
        hf_dataset_id: str = None,
    ):
        """主入口：按 dataset_name 分发到对应加载器。
        优先尝试 local JSON（./data/{name}.json），其次 HF 加载，最后 dummy。
        """
        # 1) 优先 local JSON
        local_path = os.path.join(self.data_dir, f"{dataset_name}.json")
        if os.path.exists(local_path):
            print(f"[DataManager] 加载本地 {local_path} ...")
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = data[:total_samples]
            print(f"[DataManager] 本地数据加载了 {len(data)} 条")
            return data

        # 2) HF 加载
        if dataset_name == "dummy":
            return self._generate_dummy(total_samples)
        if dataset_name == "healthcaremagic":
            return self._load_healthcaremagic(hf_dataset_id, total_samples)
        if dataset_name == "agnews":
            return self._load_agnews(hf_dataset_id, total_samples)
        if dataset_name == "naturalquestions":
            return self._load_naturalquestions(hf_dataset_id, total_samples)
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ------------------------------------------------------------------ #
    # Dummy（dev/CI 用）
    # ------------------------------------------------------------------ #

    def _generate_dummy(self, total_samples: int):
        data_file = os.path.join(self.data_dir, "dataset.json")
        if os.path.exists(data_file):
            print(f"[DataManager] 从 {data_file} 加载数据...")
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"[DataManager] 生成 {total_samples} 条 dummy 数据...")
        dummy_data = []
        for i in range(total_samples):
            dummy_data.append({
                "id": i,
                "query":  f"Question number {i} about topic {i % 50}",
                "answer": f"Ground truth response {i} with details for topic {i % 50}.",
            })
        return dummy_data

    # ------------------------------------------------------------------ #
    # 真实数据集加载器（论文 §5.1）
    # ------------------------------------------------------------------ #

    def _load_healthcaremagic(self, hf_dataset_id, total_samples):
        """
        HealthCareMagic（论文 §5.1.1）：医疗对话问答，约 11 万样本。
        默认 HF 路径: "lukasellinger/HealthCareMagic-100k-HF"
        字段: 'input' (病人问题) + 'output' (医生回答)
        """
        from datasets import load_dataset
        hf_dataset_id = hf_dataset_id or "lukasellinger/HealthCareMagic-100k-HF"
        print(f"[DataManager] 加载 HealthCareMagic from {hf_dataset_id} ...")
        ds = load_dataset(hf_dataset_id, split="train")
        data = []
        for i, row in enumerate(ds):
            if len(data) >= total_samples:
                break
            q, a = row.get("input", ""), row.get("output", "")
            if not q or not a:
                continue
            data.append({"id": i, "query": q.strip(), "answer": a.strip()})
        print(f"[DataManager] HealthCareMagic 加载了 {len(data)} 条")
        return data

    def _load_agnews(self, hf_dataset_id, total_samples):
        """
        AgNews（论文 §5.1.2）：新闻文本数据集，**前 10 词 = query，剩余 = answer**。
        默认 HF 路径: "fancyzhx/ag_news"
        """
        from datasets import load_dataset
        hf_dataset_id = hf_dataset_id or "fancyzhx/ag_news"
        print(f"[DataManager] 加载 AgNews from {hf_dataset_id} ...")
        ds = load_dataset(hf_dataset_id, split="train")
        data = []
        i = 0
        for row in ds:
            if len(data) >= total_samples:
                break
            text = row.get("text", "")
            words = text.split()
            if len(words) < 11:
                continue
            data.append({
                "id": i,
                "query":  " ".join(words[:10]),
                "answer": " ".join(words[10:]),
            })
            i += 1
        print(f"[DataManager] AgNews 加载了 {len(data)} 条")
        return data

    def _load_naturalquestions(self, hf_dataset_id, total_samples):
        """
        NaturalQuestions（论文 §5.1.3）：通用大规模问答。
        默认 HF 路径: "google-research-datasets/nq_open"
        字段: 'question' + 'answer' (answer 是 list[str])
        """
        from datasets import load_dataset
        hf_dataset_id = hf_dataset_id or "google-research-datasets/nq_open"
        print(f"[DataManager] 加载 NaturalQuestions from {hf_dataset_id} ...")
        ds = load_dataset(hf_dataset_id, split="train")
        data = []
        for i, row in enumerate(ds):
            if len(data) >= total_samples:
                break
            q = row.get("question", "")
            a = row.get("answer", [""])[0] if row.get("answer") else ""
            if not q or not a:
                continue
            data.append({"id": i, "query": q.strip(), "answer": a.strip()})
        print(f"[DataManager] NaturalQuestions 加载了 {len(data)} 条")
        return data

    # ------------------------------------------------------------------ #
    # 切分（保持原签名）
    # ------------------------------------------------------------------ #

    def split_data(
        self,
        data: list,
        target_kb_size: int = 8000,
        eval_members: int = 1000,
        eval_non_members: int = 1000,
        aux_members: int = 100,
        aux_non_members: int = 100,
    ):
        """按论文比例切分。"""
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
