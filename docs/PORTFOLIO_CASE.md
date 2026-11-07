# RewindAI — Product case / 产品案例

## Problem / 问题

I started from a learning problem: students may need to revisit one explanation buried in a long recording. Transcript search can locate words, but verifying an AI-generated answer also requires the original timestamp and context. This is my project hypothesis, not a finding from completed user research.

学生复习时可能需要从长录像中找到一段解释。字幕搜索能定位文字，但判断 AI 回答是否可靠仍需要回看原视频和上下文。这是项目假设，并非已经完成的用户研究结论。

## Product / 产品

I built RewindAI as a local video-learning workspace: users import an authorized recording with timed subtitles, search or ask a question, inspect candidate excerpts and timestamps, revisit the source, then save moments and notes for review. Local review status and feedback support the study workflow. An experimental visual route ranks sampled frames and can fuse those rankings with subtitle candidates; it does not generate a visually grounded answer.

我开发了 RewindAI 本地视频学习工作台：用户导入有权使用的视频与带时间戳的字幕，搜索或提问，查看候选片段并跳回原视频，再收藏片段和记录笔记。复习状态和反馈保存在本机。实验性画面检索可与字幕候选进行排名融合，但尚不能根据画面生成有视觉依据的回答。

## Design decisions / 设计取舍

- **Evidence before confidence.** The workspace shows source timestamps, neighboring subtitle context and optional midpoint frames. A retrieved candidate is not proof of factual support.
- **Local-first processing.** The subtitle, Qwen3-Embedding, CLIP and GGUF answer paths use local files and CPU inference. Private lectures, indexes, questions and results stay local. Indexing does not train model weights.
- **Explicit uncertainty.** Transcript search can abstain at its configured threshold. The experimental visual route has no validated rejection threshold and can return candidates even for absent content.
- **Evaluation before claims.** Compare fixed configurations on the same questions, preserve video-level splits and review answers against source recordings. The provided review tooling helps collect judgments but cannot establish their independence.

- **优先展示依据。** 展示时间戳、相邻字幕和可选的片段中点画面；检索到的候选不等于事实已经得到证实。
- **本地优先。** 字幕处理、Qwen3-Embedding、CLIP 和 GGUF 回答均使用本地文件和 CPU；私人录像、索引、题目和结果留在本机。建立索引不等于训练模型权重。
- **明确不确定性。** 字幕检索可以依据配置阈值不返回结果；实验性画面检索没有经过验证的拒答阈值，即使不存在匹配内容也可能返回候选。
- **先评估再写成绩。** 在同一批问题上比较固定配置，按视频划分数据，并对照录像核对回答。工具可以收集人工判断，但不能证明评审独立。

## Evidence so far / 已有证据

I evaluated Qwen3 retrieval on 151 eligible questions over provided full transcripts from a previously explored public-video partition. The top three candidates overlapped the annotated time window for **108/151 questions**; micro time-overlap precision, recall and F1 were **0.248, 0.389 and 0.303**. These are temporal measures rather than answer correctness. The run excluded 77 invalid transcripts and did not independently review video or audio. See [evaluation details](INITIAL_EVALUATION.md). I do not claim that this represents course-answer accuracy or a completed user study.

我使用先前探索过的公开视频分区，评估 Qwen3 对 151 道有效问题的完整字幕检索。前 3 个候选中，**108/151** 道题与参考时间范围重合；时间重合微平均 Precision、Recall、F1 分别为 **0.248、0.389、0.303**。这些是时间定位指标，不是回答正确率。实验排除了 77 条无效字幕，且未独立核验视频或音频；详见[评估记录](INITIAL_EVALUATION.md)。我不会将其作为课程回答准确率或已完成用户研究的证明。

## Two-minute demonstration storyboard / 两分钟演示提纲

This is a recording plan, not a claim that a demonstration video has already been recorded. Use the [public sample](PUBLIC_SAMPLE_ZH.md), not restricted course recordings; the linked Wikimedia page identifies the author and licence requirements for shared adaptations.

以下是录制计划，不代表演示视频已录制。请使用[公开示例](PUBLIC_SAMPLE_ZH.md)，不要展示受限课程录像；如需分享改编素材，按链接中的作者和许可要求处理。

| Time | Show / 展示 | Say / 说明 |
| --- | --- | --- |
| 0:00–0:20 | Project and public video / 项目与公开视频 | Finding and verifying an explanation in a long recording / 在长录像中查找并核验解释 |
| 0:20–0:55 | Search, candidate timestamps, playback / 搜索、时间戳和回看 | Candidates lead back to the source; verify the context / 候选片段帮助回到原文，需要核验上下文 |
| 0:55–1:20 | Save a moment, add a note, mark reviewed / 收藏、备注并标记已复习 | A concrete study workflow that persists locally / 展示本机可持续使用的复习流程 |
| 1:20–1:40 | Optional visual candidates / 可选画面候选 | CLIP ranks sampled frames; candidates can be irrelevant / CLIP 对采样画面排序，结果可能不相关 |
| 1:40–2:00 | Evaluation notes and next step / 评测说明和下一步 | Report only measured time-overlap metrics; independently review course questions before claiming answer F1 / 只展示已测的时间重叠指标，课程回答 F1 需独立核验 |
