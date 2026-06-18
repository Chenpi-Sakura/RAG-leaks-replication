"""
结果汇总工具：扫描 results/ 下所有 aggregate_metrics.json，
输出论文表格（markdown 格式）：dataset × LLM 矩阵，含 DC-MIA vs S-MIA 对照。

用法：
    python summarize.py                    # 扫描默认 ./results
    python summarize.py --root ./results  # 指定根目录
    python summarize.py --out summary.md  # 写到文件
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_all(root: Path):
    """
    扫描 root 下所有 aggregate_metrics.json，解析为
    {exp_name: {"dc_mia": {...}, "smia": {...}, "config": {...}} 列表
    """
    runs = defaultdict(list)  # exp_name -> [metrics_dict, ...]
    for p in root.rglob("aggregate_metrics.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 拿最近一层目录名当 exp 标识
        exp = p.parent.name
        # 如果聚合里只有 n_seeds 数字，标记为 multi-seed；否则是 single
        if isinstance(data, dict) and "n_seeds" in data and "dc_mia" in data:
            runs[exp].append(data)
    return runs


def fmt(mean, std):
    if mean is None:
        return "—"
    if std and std > 0:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def render_table(runs, out_path: Path = None) -> str:
    """
    输出 markdown 表格：行=exp_name，列=DC-MIA AUC / DC-MIA TPR / S-MIA AUC / S-MIA TPR / delta AUC
    """
    lines = ["# DC-MIA vs S-MIA 实验结果汇总\n"]
    if not runs:
        lines.append("_暂无实验结果。_")
        text = "\n".join(lines)
        if out_path:
            out_path.write_text(text, encoding="utf-8")
        return text

    for exp, run_list in sorted(runs.items()):
        # 一个实验可能跑多次（sweep）— 列出每个 run + 平均
        lines.append(f"## Experiment: `{exp}`\n")
        lines.append("| Run | DC-MIA AUC | DC-MIA TPR@1%FPR | S-MIA AUC | S-MIA TPR@1%FPR | Δ AUC |")
        lines.append("|---|---|---|---|---|---|")

        all_dc_auc, all_dc_tpr, all_s_auc, all_s_tpr = [], [], [], []
        for run in run_list:
            dc = run.get("dc_mia", {})
            sm = run.get("smia", {})
            n_seeds = run.get("n_seeds", 1)

            if n_seeds > 1 and "auc" in dc and isinstance(dc["auc"], dict):
                dc_auc = fmt(dc["auc"]["mean"], dc["auc"]["std"])
                dc_tpr = fmt(dc["tpr_at_1_fpr"]["mean"], dc["tpr_at_1_fpr"]["std"])
                s_auc  = fmt(sm["auc"]["mean"], sm["auc"]["std"])
                s_tpr  = fmt(sm["tpr_at_1_fpr"]["mean"], sm["tpr_at_1_fpr"]["std"])
            else:
                # single seed 模式
                dc_auc = fmt(dc.get("auc"), 0.0)
                dc_tpr = fmt(dc.get("tpr_at_1_fpr"), 0.0)
                s_auc  = fmt(sm.get("auc"), 0.0)
                s_tpr  = fmt(sm.get("tpr_at_1_fpr"), 0.0)

            delta = ""
            try:
                if isinstance(dc.get("auc"), dict) and isinstance(sm.get("auc"), dict):
                    delta = f"+{dc['auc']['mean'] - sm['auc']['mean']:.4f}"
                elif dc.get("auc") is not None and sm.get("auc") is not None:
                    delta = f"+{dc['auc'] - sm['auc']:.4f}"
            except Exception:
                pass

            lines.append(f"| n={n_seeds} | {dc_auc} | {dc_tpr} | {s_auc} | {s_tpr} | {delta} |")
            # 累计
            try:
                if isinstance(dc.get("auc"), dict):
                    all_dc_auc.append(dc["auc"]["mean"])
                    all_dc_tpr.append(dc["tpr_at_1_fpr"]["mean"])
                    all_s_auc.append(sm["auc"]["mean"])
                    all_s_tpr.append(sm["tpr_at_1_fpr"]["mean"])
                else:
                    all_dc_auc.append(dc.get("auc"))
                    all_dc_tpr.append(dc.get("tpr_at_1_fpr"))
                    all_s_auc.append(sm.get("auc"))
                    all_s_tpr.append(sm.get("tpr_at_1_fpr"))
            except Exception:
                pass

        if len(run_list) > 1 and all_dc_auc:
            import numpy as np
            lines.append("")
            lines.append(f"**Sweep 平均** ({len(run_list)} runs):  "
                         f"DC-MIA AUC = {np.mean(all_dc_auc):.4f},  "
                         f"S-MIA AUC = {np.mean(all_s_auc):.4f}")
        lines.append("")

    # 全局汇总
    if len(runs) > 1:
        lines.append("## 全局对比\n")
        lines.append("| Experiment | DC-MIA AUC | S-MIA AUC | Δ |")
        lines.append("|---|---|---|---|")
        for exp, run_list in sorted(runs.items()):
            # 取第一个 multi-seed run 或第一个
            for r in run_list:
                dc = r.get("dc_mia", {})
                sm = r.get("smia", {})
                dc_a = dc.get("auc", {}).get("mean") if isinstance(dc.get("auc"), dict) else dc.get("auc")
                sm_a = sm.get("auc", {}).get("mean") if isinstance(sm.get("auc"), dict) else sm.get("auc")
                if dc_a is not None and sm_a is not None:
                    lines.append(f"| {exp} | {dc_a:.4f} | {sm_a:.4f} | +{dc_a - sm_a:.4f} |")
                    break

    text = "\n".join(lines)
    if out_path:
        out_path.write_text(text, encoding="utf-8")
        print(f"[Summarize] 写入 {out_path}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./results", help="结果根目录")
    parser.add_argument("--out", default=None, help="输出 markdown 文件（默认 stdout）")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"结果目录不存在: {root}")
    runs = load_all(root)
    text = render_table(runs, out_path=Path(args.out) if args.out else None)
    if not args.out:
        print(text)


if __name__ == "__main__":
    main()
