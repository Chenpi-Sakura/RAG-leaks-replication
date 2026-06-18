#!/usr/bin/env bash
# 一键下载并转换论文 3 个数据集到本地 JSON（绕开 HF 镜像库缓存问题）
# 输出到 ./data/{name}.json，data_manager 优先读本地
#
# 用法：bash download_all.sh
set -e

mkdir -p data
cd "$(dirname "$0")"

echo "=== [1/3] HealthCareMagic (约 200MB) ==="
curl -L -o data/HealthCareMagic-100k.json \
  "https://hf-mirror.com/datasets/wangrongsheng/HealthCareMagic-100k-en/resolve/main/HealthCareMagic-100k.json"
python -c "
import json
with open('data/HealthCareMagic-100k.json', encoding='utf-8') as f:
    raw = json.load(f)
out = [{'id': i, 'query': r['input'].strip(), 'answer': r['output'].strip()}
       for i, r in enumerate(raw) if r.get('input') and r.get('output')]
json.dump(out, open('data/healthcaremagic.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'  → data/healthcaremagic.json: {len(out)} samples')
"
rm -f data/HealthCareMagic-100k.json

echo ""
echo "=== [2/3] AgNews (论文 §5.1.2: 前 10 词 = query) ==="
curl -L -o data/agnews_train.parquet \
  "https://hf-mirror.com/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet"
python -c "
import pandas as pd, json
df = pd.read_parquet('data/agnews_train.parquet')
out = []
for i, row in enumerate(df.itertuples(index=False)):
    words = str(row.text).split()
    if len(words) < 11: continue
    out.append({'id': i, 'query': ' '.join(words[:10]), 'answer': ' '.join(words[10:])})
json.dump(out, open('data/agnews.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'  → data/agnews.json: {len(out)} samples')
"
rm -f data/agnews_train.parquet

echo ""
echo "=== [3/3] NaturalQuestions ==="
curl -L -o data/nq_train.parquet \
  "https://hf-mirror.com/datasets/google-research-datasets/nq_open/resolve/main/nq_open/train-00000-of-00001.parquet"
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
print(f'  → data/naturalquestions.json: {len(out)} samples')
"
rm -f data/nq_train.parquet

echo ""
echo "=== 完成！data/ 目录： ==="
ls -lh data/
