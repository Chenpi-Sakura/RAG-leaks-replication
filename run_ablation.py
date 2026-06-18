"""
论文消融实验跑批脚本（PR 3）。

用法：
    # 跑 LLM 跨模型消融（论文 §6.2.4）
    python run_ablation.py \
        --base configs/llama3-8b-healthcaremagic.yaml \
        --sweep llm=llama3-8b,mistral-7b,glm-4-9b

    # 跑 Top-k 消融（论文 §6.2.2）
    python run_ablation.py \
        --base configs/llama3-8b-healthcaremagic.yaml \
        --sweep rag.top_k=1,2,4,8

    # 跑检索器消融（论文 §6.2.3）
    python run_ablation.py \
        --base configs/llama3-8b-healthcaremagic.yaml \
        --sweep rag.embedding_model=minilm,bge,bm25,ideal

    # 复合 sweep：m 和 metric 同时扫（笛卡尔积）
    python run_ablation.py \
        --base configs/llama3-8b-healthcaremagic.yaml \
        --sweep attack.m=2,8,16 \
        --sweep attack.metric=cosine,rouge2

LLM 切换通过设置环境变量 LLM_KIND / LLM_PATH / LLM_BASE_URL / LLM_API_KEY，
或直接修改 .env 后跑。

每次 sweep 写盘到 results/sweep_{timestamp}/{sweep_name}/ 子目录。
汇总用 summarize.py 一键生成 paper 表格。
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_overrides(sweep_args):
    """
    把 --sweep field=v1,v2,v3 解析为 [{field: v1}, {field: v2}, {field: v3}]。
    多个 --sweep 形成笛卡尔积。
    """
    if not sweep_args:
        return [{}]
    parsed = []
    for s in sweep_args:
        if "=" not in s:
            raise ValueError(f"--sweep 格式应为 field=v1,v2,...: {s}")
        field, values = s.split("=", 1)
        values = [v.strip() for v in values.split(",") if v.strip()]
        parsed.append([(field, v) for v in values])
    return [dict(items) for items in itertools.product(*parsed)]


def write_overridden_yaml(base_yaml: Path, overrides: dict, out_path: Path):
    """读 base yaml，覆盖指定字段，写到 out_path。"""
    import yaml
    raw = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
    for k, v in overrides.items():
        # 支持 "attack.m" 这种点分字段
        parts = k.split(".")
        node = raw
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    out_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_one(base_yaml: Path, overrides: dict, sweep_root: Path, dry_run: bool = False) -> dict:
    """跑一次实验；返回 (name, exit_code, duration, output_dir)。"""
    name = "__".join(f"{k.replace('.', '_')}={v}" for k, v in overrides.items()) or "base"
    safe_name = name.replace("/", "_").replace(":", "_")
    out_dir = sweep_root / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_yaml = sweep_root / f"_tmp_{safe_name}.yaml"
    write_overridden_yaml(base_yaml, overrides, tmp_yaml)

    print(f"\n[{name}] starting...")
    t0 = time.time()
    if dry_run:
        print(f"  dry-run: would run main.py --config {tmp_yaml}")
        return {"name": name, "exit_code": 0, "duration": 0.0, "out_dir": out_dir}

    proc = subprocess.run(
        [sys.executable, "main.py", "--config", str(tmp_yaml)],
        capture_output=True, text=True,
        # Windows 终端默认 GBK 会让 print(中文) 崩；强制 UTF-8
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - t0
    # 写日志
    (out_dir / "run.log").write_text(
        f"=== overrides ===\n{overrides}\n\n=== stdout ===\n{proc.stdout}\n\n=== stderr ===\n{proc.stderr}\n",
        encoding="utf-8",
    )
    print(f"  exit={proc.returncode}  duration={elapsed:.1f}s  log={out_dir}/run.log")
    return {"name": name, "exit_code": proc.returncode, "duration": elapsed, "out_dir": out_dir}


def main():
    parser = argparse.ArgumentParser(description="论文消融实验 sweep 跑批")
    parser.add_argument("--base", required=True, help="base YAML config path")
    parser.add_argument("--sweep", action="append", default=[],
                        help='field=v1,v2,...  (可多次；多次时笛卡尔积)')
    parser.add_argument("--dry-run", action="store_true", help="只生成 override yaml，不跑")
    parser.add_argument("--out-root", default="./results/sweeps",
                        help="sweep 输出根目录")
    args = parser.parse_args()

    base_yaml = Path(args.base)
    if not base_yaml.exists():
        raise SystemExit(f"base yaml 不存在: {base_yaml}")

    overrides_list = parse_overrides(args.sweep)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(args.out_root) / f"sweep_{ts}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    print(f"=== Sweep: {len(overrides_list)} 组 ===")
    print(f"Base: {base_yaml}")
    print(f"Output: {sweep_root}")

    # 把 sweep 配置也存一份
    import yaml
    (sweep_root / "sweep_config.yaml").write_text(
        yaml.safe_dump({
            "base": str(base_yaml),
            "sweeps": [args.sweep],
            "n_runs": len(overrides_list),
        }, allow_unicode=True),
        encoding="utf-8",
    )

    results = []
    for i, ov in enumerate(overrides_list, 1):
        print(f"\n--- [{i}/{len(overrides_list)}] ---")
        r = run_one(base_yaml, ov, sweep_root, dry_run=args.dry_run)
        results.append(r)

    # sweep 汇总
    summary = sweep_root / "sweep_summary.csv"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("name,exit_code,duration_seconds,out_dir\n")
        for r in results:
            f.write(f"{r['name']},{r['exit_code']},{r['duration']:.1f},{r['out_dir']}\n")

    ok = sum(1 for r in results if r["exit_code"] == 0)
    print(f"\n=== Sweep done: {ok}/{len(results)} 成功 ===")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
