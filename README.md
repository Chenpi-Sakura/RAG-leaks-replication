# RAG-leaks: 难度校准成员推断攻击 (DC-MIA) 工程复现

本项目是对论文《RAG-leaks: difficulty-calibrated membership inference attacks on retrieval-augmented generation》的非官方工程复现。

## 💡 这篇论文带来的"新启发"与核心思想

在阅读本项目的代码前，强烈建议先理解这篇论文在理论上打破了什么固有认知：

### 1. 过去的做法与误区：直接画等号
在过去的 RAG 隐私攻击（MIA）研究中，大家普遍默认一个直觉：**如果 RAG 系统生成的答案与真实答案的相似度 (Similarity) 越高，就说明该样本越有可能是 RAG 知识库里的"成员" (Member)。**
因此，之前的攻击方法直接定一个相似度阈值：相似度 > 0.8 就是成员，否则就是非成员。

### 2. 论文的洞察：相似度 ≠ 成员状态，相似度 = 回复难度
论文指出，把相似度直接等同于成员身份是**极其粗糙且有缺陷的**。
因为，一个 RAG 系统的"回复难度"实际上由两个部分组成：
1. **样本多样性 (Sample Diversity)**：这个问题本身对大模型来说是不是常识？是不是本来就很容易回答？
2. **成员状态 (Membership Status)**：知识库里有没有相关的背景信息？

**导致的问题**：一个本来就很简单的问题，就算它**不在**知识库里（非成员），大模型也能答得很好，拿到 0.85 的高相似度；相反，一个很难的问题，就算它**在**知识库里（成员），大模型也可能回答得有瑕疵，只拿到 0.85 的相似度。这就导致在 `0.5 ~ 0.9` 这个相似度区间内，成员和非成员发生了严重的"重叠 (Overlap)"，传统的基于固定阈值的攻击在这里彻底失效。

### 3. 破局之道：难度校准 (Difficulty Calibration)
为了剥离掉"样本本身到底难不难（样本多样性）"这个干扰因素，纯粹地评估"它是否在知识库里（成员状态）"，论文引入了 **难度校准 (DC-MIA)** 机制，包含两个阶段：
*   **阶段一（捡漏）**：对于相似度极高（接近 1.0）的样本，它们是非成员的概率微乎其微，直接判定为成员。
*   **阶段二（校准与似然比检验）**：对于落在混淆区间的样本 $x$，我们不去猜，而是**动态做实验**。
    *   我们在本地临时建 8 个包含了 $x$ 的影子知识库 (`inRAGs`)，和 8 个不包含 $x$ 的影子知识库 (`outRAGs`)。
    *   分别去问这些影子 RAG，拿到两组相似度得分，并拟合成两个正态分布曲线 $\mathcal{N}_{in}$ 和 $\mathcal{N}_{out}$。
    *   通过对比目标系统真实的相似度落在这两条曲线上的概率（**似然比**），我们就"抵消"了样本自身的难度，得到了一个纯粹反映它是否在知识库里的**校准得分 (Calibrated Score)**。

---

## 🛠️ 工程架构说明

本工程针对工程化落地做出了以下设计优化：
1. **纯内存 RAG 检索**：为了支撑极端耗时的影子知识库动态生成（每个测试样本都需要重新构建 16 个知识库），系统底层移除了传统数据库（如 pgvector），采用完全基于内存的 **FAISS** 实现，以提供极致的建库和检索速度。
2. **热切换 LLM 服务 (`llm_service.py`)**：4 个 LLM 后端（echo / openai 兼容 / vllm-server / vllm-local）通过 `BaseLLM` ABC 抽象，`build_llm_from_env()` 工厂从 `.env` 自动装配。
3. **可切换 Retriever (`retriever.py`)**：4 个检索器（minilm / bge / bm25 / ideal）通过 `BaseRetriever` ABC 抽象，对应论文 §5.2 默认 + §6 消融。
4. **配置化实验 (`config.py` + `configs/`)**：所有实验参数走 YAML 配置文件，命令行只接 `--config <path>` 切换。
5. **多 seed 跑 + 聚合统计 (`main.py`)**：默认 5 个 seed 跑均值±标准差，输出 `aggregate_metrics.json` 可直接贴论文表格。
6. **本地化数据流 (`data_manager.py`)**：自动读 `./data/{name}.json`（手动下载到本地），企业内网环境零外部依赖。

---

## 🚀 完整端到端示例（从 0 到跑出 AUC 数字）

按顺序执行，**全套下来约 30-60 分钟**（含 vLLM 装包；不计跑实验时间）。

### Step 1：创建环境（conda 推荐）

```bash
# 用 conda / mamba / micromamba 都行；这里用 conda
conda create -n rag-leaks python=3.11 -y
conda activate rag-leaks

# ⚠️ 关键：先装 vllm（拉它兼容的 torch/transformers），
#          再装其他（适配已装的 torch，避免版本冲突）
pip install vllm
pip install -r requirements.txt --no-deps

# --no-deps 阻止 pip 重新解析 torch 依赖
# 装不上时切阿里云镜像：
#   pip install -i https://mirrors.aliyun.com/pypi/simple/ vllm
#   pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt --no-deps

# 验证环境
python -c "import torch, transformers, sentence_transformers, vllm; print('all OK')"
```

> **没 GPU 也能跑**：`pip install vllm` 仍会装，但 vLLM 跑不起来。把 `.env` 里 `LLM_KIND` 设为 `echo` 或 `openai` 即可（详见 Step 4）。

### Step 2：克隆 + 装依赖

```bash
git clone https://github.com/Chenpi-Sakura/RAG-leaks-replication.git
cd RAG-leaks-replication
pip install vllm
pip install -r requirements.txt --no-deps
```

### Step 3：下载数据集（从镜像，手动一次）

论文 §5.1 用的 3 个数据集，从 **hf-mirror.com** 镜像下载（国内直连，无 GFW 问题）：

#### 3.1 HealthCareMagic（医疗对话，约 11 万样本）

```bash
mkdir -p data

# 下载（约 200MB，单文件 JSON 数组）
curl -L -o data/HealthCareMagic-100k.json \
  "https://hf-mirror.com/datasets/wangrongsheng/HealthCareMagic-100k-en/resolve/main/HealthCareMagic-100k.json"
```

转格式（论文里我们要的是 `query` / `answer` 两个字段）：

```bash
python -c "
import json
with open('data/HealthCareMagic-100k.json', encoding='utf-8') as f:
    raw = json.load(f)
out = [{'id': i, 'query': r['input'].strip(), 'answer': r['output'].strip()}
       for i, r in enumerate(raw) if r.get('input') and r.get('output')]
json.dump(out, open('data/healthcaremagic.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'Saved {len(out)} samples to data/healthcaremagic.json')
"
```

#### 3.2 AgNews（新闻文本，论文 §5.1.2：前 10 词 = query，剩余 = answer）

```bash
curl -L -o data/train-00000-of-00001.parquet \
  "https://hf-mirror.com/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet"
```

转格式（用我们代码的 AgNews 切分逻辑：前 10 词 / 剩余）：

```bash
python -c "
import pandas as pd, json
df = pd.read_parquet('data/train-00000-of-00001.parquet')
out = []
for i, row in enumerate(df.itertuples(index=False)):
    words = str(row.text).split()
    if len(words) < 11: continue
    out.append({'id': i, 'query': ' '.join(words[:10]), 'answer': ' '.join(words[10:])})
json.dump(out, open('data/agnews.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'Saved {len(out)} samples to data/agnews.json')
rm data/train-00000-of-00001.parquet
"
```

#### 3.3 NaturalQuestions（开放域 QA）

```bash
curl -L -o data/nq_train.parquet \
  "https://hf-mirror.com/datasets/google-research-datasets/nq_open/resolve/main/nq_open/train-00000-of-00001.parquet"
```

转格式：

```bash
python -c "
import pandas as pd, json
df = pd.read_parquet('data/nq_train.parquet')
out = []
for i, row in enumerate(df.itertuples(index=False)):
    q = str(row.question)
    a = row.answer[0] if hasattr(row, 'answer') and len(row.answer) > 0 else ''
    if not q.strip() or not str(a).strip(): continue
    out.append({'id': i, 'query': q.strip(), 'answer': str(a).strip()})
json.dump(out, open('data/naturalquestions.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'Saved {len(out)} samples to data/naturalquestions.json')
rm data/nq_train.parquet
"
```

#### 一键脚本（懒人版）

不想一步一步跑？把上面 3 段拼成一个 `download_all.sh`：

```bash
bash download_all.sh   # 文件已附在仓库根目录
```

加载优先级（`data_manager.py`）：
1. **本地 JSON** `./data/{dataset_name}.json` ← 上面下载的
2. HF Hub 加载（如果本地没有）
3. Dummy 假数据（仅 `dataset_name=dummy`）

### Step 4：配置 .env

```bash
cp .env.example .env
# 编辑 .env，至少改一项：
```

| 你的硬件 | `.env` 必设 | `configs/*.yaml` 必设 |
|---|---|---|
| **没 GPU** | `LLM_KIND=echo` | （yaml 默认 OK） |
| **有 GPU + 远程 API** | `LLM_KIND=openai` + `LLM_API_KEY=sk-...` | （yaml 默认 OK） |
| **有 GPU + 本地 vLLM** | `LLM_KIND=vllm-local` + `LLM_PATH=meta-llama/Meta-Llama-3-8B-Instruct` | （yaml 默认 OK） |

> 详细的 GPU 显存预算见 [⚡ GPU 配置](#-gpu-配置) 章节。

### Step 5：跑实验

```bash
# Windows 终端要加 UTF-8 前缀（避免中文 print 崩）
export PYTHONIOENCODING=utf-8

# 跑论文主表第一行（Llama-3-8B × HealthCareMagic）
python main.py --config configs/llama3-8b-healthcaremagic.yaml

# 输出在 results/{exp_name}_{timestamp}/ 下：
#   aggregate_metrics.json    ← ★ 论文表格直接复制 mean ± std
#   seeds_summary.csv         ← 每 seed 单独一行
#   seed_0/metrics.json
#   seed_0/per_sample_scores.csv
#   config_used.yaml          ← 复现用的配置
```

### Step 6：出 paper-ready 表格

```bash
# 扫描 results/ 下所有 aggregate_metrics.json → 1 张 markdown 表
python summarize.py --out summary.md
cat summary.md
```

### Step 7：跑消融实验

```bash
# §6.2.1 Phase-2-only 消融
python main.py --config configs/llama3-8b-healthcaremagic.yaml
# yaml 里改 attack.skip_phase_1: true 再跑

# §6.2.3 检索器消融（minilm / bge / bm25 / ideal 一键对比）
python run_ablation.py --base configs/llama3-8b-healthcaremagic.yaml \
  --sweep "rag.embedding_model=minilm,bge,bm25,ideal"
# 输出在 results/sweeps/sweep_{ts}/  下；用 summarize.py 汇总

# §7.2 ROUGE 指标（vs 默认 cosine）
# yaml 里改 attack.metric: rouge2 再跑
```

---

## ⚡ GPU 配置

不同硬件显存预算：

| 硬件 | LLM_GPU_MEM_UTIL | `rag.device` (yaml) | 说明 |
|---|---|---|---|
| 单卡 24GB 跑 8B | 0.85 | auto | vLLM 拿 20GB，剩 4GB 给 embedder |
| 单卡 16GB 跑 8B | 0.80 | auto | vLLM 拿 13GB，剩 3GB |
| 单卡 24GB 跑 70B | 0.92 | **cpu** | 70B 几乎吃满，embedder 必须 CPU |
| 多卡 2×24GB | 0.85 | cuda:1 | vLLM 锁第 0 卡，embedder 锁第 1 卡 |
| 没 GPU | N/A | cpu | vLLM 跑不起来，要换 OpenAICompatLLM |

> `LLM_GPU_MEM_UTIL` 在 `.env` 里设（默认 0.85）。
> `rag.device` 在 `configs/*.yaml` 里设（默认 `auto`）。

---

## 📦 数据集加载规则

`data_manager.py` 按以下优先级加载：

1. **本地 JSON**（推荐）：`./data/{dataset_name}.json`，格式 `[{"id", "query", "answer"}, ...]`
2. **HF Hub**（如果本地没有）
3. **Dummy 假数据**（仅 `dataset_name=dummy`）

### 推荐的 HF ID（如果在镜像上找不到，可去 HF 官方搜）

| 数据集 | 推荐 ID | 字段 |
|---|---|---|
| HealthCareMagic | `wangrongsheng/HealthCareMagic-100k-en` | input / output |
| AgNews | `fancyzhx/ag_news` | text（论文 §5.1 切前 10 词） |
| NaturalQuestions | `google-research-datasets/nq_open` | question / answer |

> **注意**：`huggingface_hub` 和 `datasets` 库的 endpoint 在 import 时就冻结，`HF_ENDPOINT` 环境变量经常不生效。**最稳的方式就是 wget/curl 下到 `./data/`**（如 Step 3 所示）。

---

## 🧪 验证（确认安装正确）

```bash
# 1. 导入链路
python -c "from llm_service import build_llm_from_env; from retriever import build_retriever_from_config; from attack_core import DCMIA; from smia import SMIA; print('imports OK')"

# 2. echo 后端（不烧钱）跑一次小实验
echo "LLM_KIND=echo" > .env
python -c "
from config import ExperimentConfig
from data_manager import DataManager
from llm_service import build_llm_from_env
from retriever import build_retriever_from_config
from rag_system import SimpleRAG
from attack_core import DCMIA
from smia import SMIA
from evaluator import Evaluator

cfg = ExperimentConfig.from_yaml('configs/llama3-8b-healthcaremagic.yaml')
print('cfg.attack.m:', cfg.attack.m)
print('cfg.dataset.target_kb_size:', cfg.dataset.target_kb_size)

dm = DataManager()
raw = dm.load_or_generate_dummy_data(total_samples=cfg.dataset.total_samples, dataset_name=cfg.dataset.name)
splits = dm.split_data(raw, target_kb_size=cfg.dataset.target_kb_size, eval_members=cfg.dataset.eval_members, eval_non_members=cfg.dataset.eval_non_members, aux_members=cfg.dataset.aux_members, aux_non_members=cfg.dataset.aux_non_members)
print(f'loaded {len(raw)} samples, split into {len(splits[\"target_kb\"])} target_kb + {len(splits[\"eval_data\"])} eval + {len(splits[\"aux_data\"])} aux')
print('ALL OK')
"
```

---

## ❓ 常见问题

**Q: `HF_ENDPOINT` 环境变量设了不生效？**
A: `huggingface_hub` 和 `datasets` 库的 endpoint 在 import 时就冻结了常量，env var 设了没生效。**最稳的解决就是 wget 下到 `./data/`**，README Step 3 有详细命令。

**Q: `pip install vllm` 装上但 vLLM 跑不起来？**
A: 多半是缺 CUDA 版 torch。先看 `python -c "import torch; print(torch.__version__)"` 是否带 `+cuXXX`，没带就要重装：
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

**Q: AUC ≈ 0.5 是 bug 吗？**
A: 看用的什么 LLM：
- `LLM_KIND=echo` → LLM 只回 prompt 末行，没用 context → AUC ≈ 0.5 是**预期**（验框架 OK）
- `LLM_KIND=openai` 远程 API → AUC 应该 > 0.7（真 LLM 有信号）
- `LLM_KIND=vllm-local` 本地真模型 → AUC 应该 > 0.7

**Q: Windows 终端中文 print 崩？**
A: 加 `PYTHONIOENCODING=utf-8` 前缀（README Step 5 有写）。

**Q: 跑太慢怎么办？**
A: 5-seed × 2000 样本 × m=8 = 32k LLM 调用 + 17× 嵌入。CPU 上一轮 ~4 分钟，GPU 上 vLLM 应 < 30 秒/seed。短期方案：减 `attack.m` 或 `seeds`；长期：装 CUDA torch + vLLM 跑 GPU。
