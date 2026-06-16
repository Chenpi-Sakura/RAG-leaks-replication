"""
实验配置 schema：定义 dataclass + YAML 加载。

使用：
    cfg = ExperimentConfig.from_yaml("configs/llama3-8b-healthcaremagic.yaml")
    print(cfg.dataset.name)         # "healthcaremagic"
    print(cfg.attack.m)            # 8

约定：环境敏感配置（API key、模型路径）走 .env；可复现实验参数走 YAML。
"""
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class DatasetConfig:
    name: str = "dummy"                    # dummy | healthcaremagic | agnews | naturalquestions
    hf_dataset_id: Optional[str] = None    # 真实数据集的 HuggingFace id；dummy 模式忽略
    total_samples: int = 12000             # dummy 模式生成多少
    target_kb_size: int = 8000
    eval_members: int = 1000
    eval_non_members: int = 1000
    aux_members: int = 100
    aux_non_members: int = 100


@dataclass
class RAGConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4
    prompt_template: str = (
        "Please answer the question based on the provided context.\n"
        "Context: {context}\nQuestion: {question}"
    )


@dataclass
class AttackConfig:
    m: int = 8
    tau_1: str = "auto"                    # "auto" 从 aux_data 扫最优；或写死 float 字符串
    tau_2: str = "auto"                    # "auto" 从混淆区扫最优
    per_sample_seed: bool = True           # 影子 RAG 采样用 sample_id 作种子，保证可复现


@dataclass
class ExperimentConfig:
    name: str = "dc-mia-repro"
    seed: int = 42                         # 单一 seed 时使用
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str = "./results"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        exp = raw.get("experiment", {})
        return cls(
            name=exp.get("name", "dc-mia-repro"),
            seed=exp.get("seed", 42),
            seeds=exp.get("seeds", [0, 1, 2, 3, 4]),
            output_dir=exp.get("output_dir", "./results"),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            rag=RAGConfig(**raw.get("rag", {})),
            attack=AttackConfig(**raw.get("attack", {})),
        )
