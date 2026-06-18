"""
DC-MIA + S-MIA 对照实验主入口。

用法：
    python main.py --config configs/llama3-8b-healthcaremagic.yaml

每个 seed 同时跑：
  - DC-MIA（论文 §4 两阶段攻击）
  - S-MIA（论文 §5.3 / §6.1 基线）
结果对比写入 aggregate_metrics.json + seeds_summary.csv。
"""
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # 必须在 import llm_service 之前


def _check_compat():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        n_gpu = torch.cuda.device_count() if cuda_ok else 0
        torch_v = torch.__version__
    except ImportError:
        torch_v, cuda_ok, n_gpu = "?", False, 0
    try:
        import transformers
        tf_v = transformers.__version__
    except ImportError:
        tf_v = "?"
    try:
        import sentence_transformers
        st_v = sentence_transformers.__version__
    except ImportError:
        st_v = "?"
    print(f"[Compat] torch={torch_v}  transformers={tf_v}  sentence-transformers={st_v}  "
          f"cuda={cuda_ok}  n_gpu={n_gpu}")
    if not cuda_ok and os.environ.get("LLM_KIND") == "vllm-local":
        print("[Compat] WARNING: LLM_KIND=vllm-local 需要 GPU，但 CUDA 不可用；"
              "请改用 vllm-server / openai / echo")


_check_compat()

from config import ExperimentConfig
from data_manager import DataManager
from llm_service import build_llm_from_env
from rag_system import SimpleRAG
from retriever import build_retriever_from_config
from attack_core import DCMIA
from smia import SMIA
from evaluator import Evaluator


def _split_mode_for(dataset_name: str) -> str:
    """不同数据集的 S-MIA split 策略不一样（论文隐含）。"""
    if dataset_name == "agnews":
        return "word"   # AgNews 按词分半（前 10 词 / 剩余）
    return "char"      # 默认按字符


def _prompt_for(cfg: ExperimentConfig) -> str:
    """AgNews 用专用 prompt 模板（论文 §5.2）。"""
    if cfg.dataset.name == "agnews":
        return cfg.rag.agnews_prompt
    return cfg.rag.prompt_template


def run_dc_mia(attacker, target_rag, eval_data, target_ids, tau_1, tau_2, m, global_seed, skip_phase_1):
    """跑 DC-MIA 攻击测试集，返回 (y_true, y_scores, y_decisions, n_llm_calls, elapsed)。"""
    y_true, y_scores, y_decisions = [], [], []
    n_llm_calls = 0
    start = time.time()
    for s in eval_data:
        if skip_phase_1:
            # §6.2.1：跳过阶段 1，直接走阶段 2
            random.seed(global_seed * 1_000_003 + s["id"])
            lr = attacker._compute_lr(target_rag, s, m=m)
            score, decision = lr, int(lr > tau_2)
            n_llm_calls += 2 * m
        else:
            score, decision = attacker.attack(target_rag, s, tau_1, tau_2, m=m, global_seed=global_seed)
            n_llm_calls += (1 + 2 * m)
        y_true.append(1 if s["id"] in target_ids else 0)
        y_scores.append(score)
        y_decisions.append(decision)
    elapsed = time.time() - start
    return y_true, y_scores, y_decisions, n_llm_calls, elapsed


def run_smia(attacker, target_rag, eval_data, target_ids, tau_s, split_mode):
    """跑 S-MIA 攻击测试集。"""
    y_true, y_scores, y_decisions = [], [], []
    start = time.time()
    for s in eval_data:
        score, _ = attacker.attack(target_rag, s, split_mode=split_mode)
        y_true.append(1 if s["id"] in target_ids else 0)
        y_scores.append(score)
        y_decisions.append(int(score > tau_s))
    elapsed = time.time() - start
    return y_true, y_scores, y_decisions, elapsed


def run_one_seed(cfg: ExperimentConfig, seed: int, out_dir: Path) -> dict:
    """跑一个 seed：DC-MIA + S-MIA 对照。"""
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

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
    retriever = build_retriever_from_config(cfg.rag.embedding_model, device=cfg.rag.device)
    retriever.warmup([d["answer"] for d in raw])
    retriever.warmup([d["query"]  for d in raw])

    # 3) 组件
    llm = build_llm_from_env()
    target_rag = SimpleRAG(
        llm_service=llm,
        retriever=retriever,
        top_k=cfg.rag.top_k,
        prompt_template=_prompt_for(cfg),
        no_rag=cfg.rag.no_rag,
    )
    target_rag.build_index(splits["target_kb"])

    # Ideal retriever 模式：注入 q2a 映射
    if retriever.kind == "ideal":
        q2a = {s["query"]: s["answer"] for s in splits["target_kb"]}
        retriever.set_q2a(q2a)
        print(f"[Main] IdealRetriever: 注入了 {len(q2a)} 个 q2a 映射")

    target_ids = {s["id"] for s in splits["target_kb"]}

    # 4) DC-MIA 阈值
    dc_attacker = DCMIA(
        llm_service=llm,
        data_pool=splits["reference_pool"],
        retriever=retriever,
        per_sample_seed=cfg.attack.per_sample_seed,
        metric=cfg.attack.metric,
    )
    if cfg.attack.skip_phase_1:
        tau_1 = None
    else:
        tau_1 = dc_attacker.phase_1_find_threshold(
            target_rag, splits["aux_data"], target_ids, tau_1_spec=cfg.attack.tau_1)
    tau_2 = dc_attacker.find_tau_2(
        target_rag, splits["aux_data"], target_ids, tau_1 if tau_1 is not None else 1.0,
        m=cfg.attack.m, global_seed=seed)
    (seed_dir / "thresholds.json").write_text(json.dumps({
        "tau_1": tau_1, "tau_2": tau_2, "m": cfg.attack.m,
        "skip_phase_1": cfg.attack.skip_phase_1,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) DC-MIA 攻击
    y_true, y_scores, y_decisions, n_dc_calls, dc_elapsed = run_dc_mia(
        dc_attacker, target_rag, splits["eval_data"], target_ids,
        tau_1, tau_2, cfg.attack.m, seed, cfg.attack.skip_phase_1)

    # 6) DC-MIA 评测 + 写盘
    dc_metrics = Evaluator.calculate_metrics(y_true, y_scores)
    dc_metrics.update({
        "tau_1": tau_1, "tau_2": tau_2, "m": cfg.attack.m,
        "n_samples": len(y_true), "seed": seed,
        "skip_phase_1": cfg.attack.skip_phase_1,
    })
    (seed_dir / "per_sample_scores.csv").write_text(
        "\n".join([f"sample_id,y_true,score,decision"] +
                  [f"{e['id']},{yt},{s:.6f},{d}" for e, yt, s, d in
                   zip(splits["eval_data"], y_true, y_scores, y_decisions)]) + "\n",
        encoding="utf-8")

    # 7) S-MIA 阈值 + 攻击
    smia = SMIA(retriever=retriever)
    split_mode = _split_mode_for(cfg.dataset.name)
    tau_s = smia.find_threshold(target_rag, splits["aux_data"], target_ids, split_mode=split_mode)
    y_true_s, y_scores_s, y_decisions_s, s_elapsed = run_smia(
        smia, target_rag, splits["eval_data"], target_ids, tau_s, split_mode)
    smia_metrics = Evaluator.calculate_metrics(y_true_s, y_scores_s)
    smia_metrics.update({"tau_s": tau_s, "split_mode": split_mode, "n_samples": len(y_true_s)})
    (seed_dir / "per_sample_scores_smia.csv").write_text(
        "\n".join([f"sample_id,y_true,score,decision"] +
                  [f"{e['id']},{yt},{s:.6f},{d}" for e, yt, s, d in
                   zip(splits["eval_data"], y_true_s, y_scores_s, y_decisions_s)]) + "\n",
        encoding="utf-8")

    # 8) 总 metrics + runtime
    (seed_dir / "metrics.json").write_text(json.dumps({
        "dc_mia": dc_metrics,
        "smia": smia_metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (seed_dir / "runtime.txt").write_text(
        f"dc_mia_elapsed_seconds={dc_elapsed:.2f}\n"
        f"dc_mia_n_llm_calls={n_dc_calls}\n"
        f"smia_elapsed_seconds={s_elapsed:.2f}\n"
        f"smia_n_llm_calls={len(y_true_s)}\n",
        encoding="utf-8",
    )
    print(f"  [seed {seed}] DC-MIA: AUC={dc_metrics['auc']:.4f} TPR@1%FPR={dc_metrics['tpr_at_1_fpr']:.4f}  "
          f"|  S-MIA: AUC={smia_metrics['auc']:.4f} TPR@1%FPR={smia_metrics['tpr_at_1_fpr']:.4f}")
    return {"seed": seed, "dc_mia": dc_metrics, "smia": smia_metrics}


def aggregate_seeds(per_seed: list, out_dir: Path) -> dict:
    """聚合 DC-MIA + S-MIA 各自的 mean ± std。"""
    def _stats(values):
        if len(values) <= 1:
            v = values[0]
            return {"mean": float(v), "std": 0.0, "min": float(v), "max": float(v), "values": values}
        return {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values, ddof=1)),
            "min":  float(np.min(values)),
            "max":  float(np.max(values)),
            "values": values,
        }

    dc_aucs = [m["dc_mia"]["auc"] for m in per_seed]
    dc_tprs = [m["dc_mia"]["tpr_at_1_fpr"] for m in per_seed]
    s_aucs  = [m["smia"]["auc"] for m in per_seed]
    s_tprs  = [m["smia"]["tpr_at_1_fpr"] for m in per_seed]

    dc_auc_stats = _stats(dc_aucs)
    dc_tpr_stats = _stats(dc_tprs)
    s_auc_stats  = _stats(s_aucs)
    s_tpr_stats  = _stats(s_tprs)

    agg = {
        "n_seeds": len(per_seed),
        "dc_mia": {"auc": dc_auc_stats, "tpr_at_1_fpr": dc_tpr_stats},
        "smia":   {"auc": s_auc_stats,  "tpr_at_1_fpr": s_tpr_stats},
        "delta": {
            "auc":  dc_auc_stats["mean"] - s_auc_stats["mean"],
            "tpr":  dc_tpr_stats["mean"] - s_tpr_stats["mean"],
        },
        "format_for_paper": (
            f"DC-MIA: AUC = {dc_auc_stats['mean']:.4f} ± {dc_auc_stats['std']:.4f}, "
            f"TPR@1%FPR = {dc_tpr_stats['mean']:.4f} ± {dc_tpr_stats['std']:.4f}  |  "
            f"S-MIA: AUC = {s_auc_stats['mean']:.4f} ± {s_auc_stats['std']:.4f}, "
            f"TPR@1%FPR = {s_tpr_stats['mean']:.4f} ± {s_tpr_stats['std']:.4f}  |  "
            f"ΔAUC = +{dc_auc_stats['mean'] - s_auc_stats['mean']:.4f}"
        ),
    }
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / "seeds_summary.csv", "w", encoding="utf-8") as f:
        f.write("seed,dc_auc,dc_tpr,smia_auc,smia_tpr,tau_1,tau_2,tau_s,dc_elapsed,smia_elapsed\n")
        for m, raw in zip(per_seed, [None] * len(per_seed)):
            rt = (out_dir / f"seed_{m['seed']}" / "runtime.txt").read_text()
            def _v(k):
                return float([l for l in rt.splitlines() if l.startswith(k + "=")][0].split("=")[1])
            tau_1 = m["dc_mia"].get("tau_1", 0) or 0
            tau_2 = m["dc_mia"]["tau_2"]
            tau_s = m["smia"]["tau_s"]
            f.write(f"{m['seed']},{m['dc_mia']['auc']:.6f},{m['dc_mia']['tpr_at_1_fpr']:.6f},"
                    f"{m['smia']['auc']:.6f},{m['smia']['tpr_at_1_fpr']:.6f},"
                    f"{tau_1:.6f},{tau_2:.6f},{tau_s:.6f},"
                    f"{_v('dc_mia_elapsed_seconds'):.2f},{_v('smia_elapsed_seconds'):.2f}\n")
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

    print(f"=== DC-MIA + S-MIA Experiment: {cfg.name} ===")
    print(f"Dataset: {cfg.dataset.name}  LLM: {os.environ.get('LLM_KIND', '?')}  "
          f"Retriever: {cfg.rag.embedding_model}  Seeds: {cfg.seeds}")
    print(f"Output: {out_dir}")

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
        print(f"DC-MIA: AUC={m['dc_mia']['auc']:.4f}  TPR@1%FPR={m['dc_mia']['tpr_at_1_fpr']:.4f}")
        print(f"S-MIA:  AUC={m['smia']['auc']:.4f}  TPR@1%FPR={m['smia']['tpr_at_1_fpr']:.4f}")


if __name__ == "__main__":
    main()
