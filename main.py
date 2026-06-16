"""
DC-MIA 实验主入口。

用法：
    python main.py --config configs/llama3-8b-healthcaremagic.yaml

环境：
    .env 管 LLM 拓扑（API key、模型路径、vllm 选择）
    configs/*.yaml 管实验参数（数据集、m、top_k、seeds、tau_1/2 等）
"""
import argparse
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # 必须在 import llm_service 之前

from config import ExperimentConfig
from data_manager import DataManager
from llm_service import build_llm_from_env
from rag_system import SimpleRAG
from retriever import build_retriever_from_config
from attack_core import DCMIA
from evaluator import Evaluator


def run_one_seed(cfg: ExperimentConfig, seed: int, out_dir: Path) -> dict:
    """跑一个 seed，返回该 seed 的 metrics dict。"""
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    # 全局随机种子（per-sample 种子在 attack() 内部另设）
    random.seed(seed)
    np.random.seed(seed)

    # 1) 数据
    dm = DataManager(data_dir="./data")
    raw = dm.load_or_generate_dummy_data(
        total_samples=cfg.dataset.total_samples,
        dataset_name=cfg.dataset.name,
        hf_dataset_id=cfg.dataset.hf_dataset_id,
    )
    splits = dm.split_data(
        raw,
        target_kb_size=cfg.dataset.target_kb_size,
        eval_members=cfg.dataset.eval_members,
        eval_non_members=cfg.dataset.eval_non_members,
        aux_members=cfg.dataset.aux_members,
        aux_non_members=cfg.dataset.aux_non_members,
    )

    # 2) Retriever（一次构造，跨 RAG 共享）
    retriever = build_retriever_from_config(cfg.rag.embedding_model)
    # Warmup 仅对 dense 类有意义；BM25/mock 自动 no-op
    retriever.warmup([d["answer"] for d in raw])
    retriever.warmup([d["query"]  for d in raw])

    # 3) 组件
    llm = build_llm_from_env()
    target_rag = SimpleRAG(
        llm_service=llm,
        retriever=retriever,
        top_k=cfg.rag.top_k,
        prompt_template=cfg.rag.prompt_template,
    )
    target_rag.build_index(splits["target_kb"])

    attacker = DCMIA(
        llm_service=llm,
        data_pool=splits["reference_pool"],
        retriever=retriever,
        per_sample_seed=cfg.attack.per_sample_seed,
    )
    target_ids = {s["id"] for s in splits["target_kb"]}

    # 3) 阈值
    tau_1 = attacker.phase_1_find_threshold(
        target_rag, splits["aux_data"], target_ids, tau_1_spec=cfg.attack.tau_1)
    tau_2 = attacker.find_tau_2(
        target_rag, splits["aux_data"], target_ids, tau_1, m=cfg.attack.m, global_seed=seed)
    (seed_dir / "thresholds.json").write_text(json.dumps({
        "tau_1": tau_1, "tau_2": tau_2, "m": cfg.attack.m,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4) 攻击测试集
    y_true, y_scores, y_decisions = [], [], []
    csv_path = seed_dir / "per_sample_scores.csv"
    with open(csv_path, "w", encoding="utf-8") as csv:
        csv.write("sample_id,y_true,score,decision\n")
        start = time.time()
        n_llm_calls = 0
        for s in splits["eval_data"]:
            score, decision = attacker.attack(target_rag, s, tau_1, tau_2, m=cfg.attack.m, global_seed=seed)
            n_llm_calls += (1 + 2 * cfg.attack.m)
            y_true.append(1 if s["id"] in target_ids else 0)
            y_scores.append(score)
            y_decisions.append(decision)
            csv.write(f"{s['id']},{y_true[-1]},{score:.6f},{decision}\n")
        elapsed = time.time() - start

    # 5) 评测 + 写盘
    metrics = Evaluator.calculate_metrics(y_true, y_scores)
    metrics.update({
        "tau_1": tau_1, "tau_2": tau_2, "m": cfg.attack.m,
        "n_samples": len(y_true), "seed": seed,
    })
    (seed_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "runtime.txt").write_text(
        f"elapsed_seconds={elapsed:.2f}\n"
        f"n_llm_calls={n_llm_calls}\n"
        f"avg_sec_per_sample={elapsed / len(y_true):.2f}\n",
        encoding="utf-8",
    )
    print(f"  [seed {seed}] AUC={metrics['auc']:.4f}  TPR@1%FPR={metrics['tpr_at_1_fpr']:.4f}")
    return metrics


def aggregate_seeds(per_seed_metrics: list, out_dir: Path) -> dict:
    """聚合多 seed 的指标，写出 mean ± std。"""
    aucs = [m["auc"] for m in per_seed_metrics]
    tprs = [m["tpr_at_1_fpr"] for m in per_seed_metrics]
    n = len(aucs)

    def _stats(xs):
        if len(xs) <= 1:
            return {"mean": float(xs[0]), "std": 0.0,
                    "min": float(xs[0]), "max": float(xs[0]), "values": xs}
        return {
            "mean": float(np.mean(xs)),
            "std":  float(np.std(xs, ddof=1)),
            "min":  float(np.min(xs)),
            "max":  float(np.max(xs)),
            "values": xs,
        }

    auc_stats = _stats(aucs)
    tpr_stats = _stats(tprs)
    agg = {
        "n_seeds": n,
        "auc": auc_stats,
        "tpr_at_1_fpr": tpr_stats,
        "format_for_paper": (
            f"AUC = {auc_stats['mean']:.4f} ± {auc_stats['std']:.4f}, "
            f"TPR@1%FPR = {tpr_stats['mean']:.4f} ± {tpr_stats['std']:.4f}"
        ),
    }
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    # seeds_summary.csv
    summary_path = out_dir / "seeds_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("seed,auc,tpr_at_1_fpr,tau_1,tau_2,elapsed_seconds\n")
        for m in per_seed_metrics:
            rt = (out_dir / f"seed_{m['seed']}" / "runtime.txt").read_text()
            elapsed = float([
                l for l in rt.splitlines() if "elapsed_seconds=" in l
            ][0].split("=")[1])
            f.write(f"{m['seed']},{m['auc']:.6f},{m['tpr_at_1_fpr']:.6f},"
                    f"{m['tau_1']:.6f},{m['tau_2']:.6f},{elapsed:.2f}\n")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / f"{cfg.name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config_used.yaml")

    print(f"=== DC-MIA Experiment: {cfg.name} ===")
    print(f"Seeds: {cfg.seeds}  Output: {out_dir}")

    per_seed = []
    for seed in cfg.seeds:
        print(f"\n[Seed {seed}] start")
        m = run_one_seed(cfg, seed, out_dir)
        per_seed.append(m)

    if len(per_seed) > 1:
        agg = aggregate_seeds(per_seed, out_dir)
        print(f"\n=== Aggregate over {agg['n_seeds']} seeds ===")
        print(agg["format_for_paper"])
    else:
        m = per_seed[0]
        print(f"\n=== Single seed result ===")
        print(f"AUC={m['auc']:.4f}  TPR@1%FPR={m['tpr_at_1_fpr']:.4f}")


if __name__ == "__main__":
    main()
