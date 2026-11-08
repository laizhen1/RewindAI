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
  --device cpu
```

结果写入：

- `artifacts/evaluation/retrieval_ablation.json`
- `artifacts/evaluation/retrieval_ablation.csv`

脚本比较 BM25、Qwen Embedding、CLIP、三种双路 RRF、三路 RRF 和
`三路 RRF + Qwen3-Reranker`，并报告宏平均 Recall@1/5/10、Hit@1/5/10、MRR、MRR@K、NDCG、平均延迟、
P95 延迟和按问题类型分组的指标。

启用 `--with-reranker` 后，同时输出 `embedding` 与 `embedding+reranker`（相同候选集的重排），以及 `all_rrf` 与 `all_rrf+reranker`。这只是实验能力，不代表已测出提升。文本索引及 Reranker 使用 `searchable_text`，可能含视觉描述，不能直接称为纯字幕基线；比较纯字幕时须先建立只包含字幕的片段和索引。CPU 是默认设备，真实模型可能很慢；运行前确认模型及数据使用许可，并确保课程数据与报告留在本机。

只报告不超过实际 `--top-k` 的截断指标。例如 `--top-k 5` 输出 Recall@5、Hit@5、MRR@5，不输出 @10。未截断的 `mrr` 是完整返回列表中第一个相关片段的倒数排名，与 `mrr@5` 不同。

运行前需使用 `scripts/build_indexes.py` 为相同片段和模型构建索引；评测会验证已有清单的片段、模型及索引文件哈希，并将清单写入本机报告。哈希匹配不能独立证明索引构建过程正确。Reranker 模型名称会记录，但此旧实验尚未固定其模型修订或依赖版本，因此不能单靠报告复现完整运行；在正式实验前应另外冻结版本与数据划分。

## 历史诊断观察（不是正式准确率）

- BM25 是新闻事实题上最强的单路基线。
- 英文 `openai/clip-vit-large-patch14` 的历史 Hit@1/5 为 5%/5%。
- `OFA-Sys/chinese-clip-vit-base-patch16` 的历史视觉单路 Hit@1/5 为
  35%/45%，三路 RRF 的历史 Hit@1/5 从 60%/90% 变为 65%/95%。
- Qwen3-Reranker 在当时的候选集上改善了 Hit@5，但未改善 Hit@1。

这些历史观察仅对应当时的 20 题自动候选集，未在本项目中重新验证，不可用于宣称
模型的真实准确率或写成正式简历指标。下一步应增加纯视觉、OCR、跨模态和无答案问题，
并继续测试动态路由、加权 RRF 或将中文查询翻译成英文后再执行图文检索。

历史脚本将上述“至少命中一个相关片段”的比例误称为 Recall@K；它实际上是 Hit@K。当前脚本将 Recall@K 定义为每题找回的相关片段数除以该题全部已标记相关片段数，再对题目取平均，并另报 Hit@K。历史数值不能直接当作新版 Recall；相关标注不完整也会高估该比例。
