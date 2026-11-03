# 中文视频问答评测

本目录包含中文新闻问题候选集和历史实验记录。根据项目当前复核流程，种子、候选和
复核队列均不可直接当作已经逐条人工核验的金标准；它们仅用于流程检查和诊断。
正式评测应先核验原视频、证据时间范围、答案和数据划分，详见
[`docs/REWINDAI_EVALUATION_ZH.md`](../../docs/REWINDAI_EVALUATION_ZH.md)。

## 数据格式

每行是一个 JSON 对象，主要字段如下：

- `question_id`: 唯一问题编号。
- `question`: 中文问题。
- `answer`: 候选答案，须经人工核验。
- `answer_aliases`: 可接受的同义答案。
- `relevant_segment_ids`: 候选相关视频片段，须经人工核验。
- `question_type`: `audio`、`visual` 或 `multimodal`。
- `evidence_start` / `evidence_end`: 证据时间范围。

## 运行实验

```bash
source env.sh
python scripts/evaluate_retrieval.py \
  --questions data/evaluation/questions.zh.seed.jsonl \
  --with-reranker \
  --device cuda
```

结果写入：

- `artifacts/evaluation/retrieval_ablation.json`
- `artifacts/evaluation/retrieval_ablation.csv`

脚本比较 BM25、Qwen Embedding、CLIP、三种双路 RRF、三路 RRF 和
`三路 RRF + Qwen3-Reranker`，并报告 Recall@1/5/10、MRR、NDCG、平均延迟、
P95 延迟和按问题类型分组的指标。

## 历史诊断观察（不是正式准确率）

- BM25 是新闻事实题上最强的单路基线。
- 英文 `openai/clip-vit-large-patch14` 对中文查询表现很差：Recall@1/5 为 5%/5%。
- `OFA-Sys/chinese-clip-vit-base-patch16` 将视觉单路 Recall@1/5 提升到
  35%/45%，并将三路 RRF Recall@1/5 从 60%/90% 提升到 65%/95%。
- Qwen3-Reranker 能改善 Recall@5，但当前未改善 Recall@1。

这些历史观察仅对应当时的 20 题自动候选集，未在本项目中重新验证，不可用于宣称
模型的真实准确率或写成正式简历指标。下一步应增加纯视觉、OCR、跨模态和无答案问题，
并继续测试动态路由、加权 RRF 或将中文查询翻译成英文后再执行图文检索。
