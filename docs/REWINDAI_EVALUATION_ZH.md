# RewindAI 课程视频评估与人工标注方案

本方案用于计划小规模试验，**不代表任何实验已完成，也不提供性能结论**。原仓库提供片段、索引、检索实验和人工复核工具；使用前请检查其具体输入、输出和限制。先确认素材和代码使用权限，课程录屏、字幕、帧图与任何可识别个人的信息均不随模板提交。

## 1. 问题与比较对象

研究问题：在相同课程视频和问题集上，加入视觉信息是否能改善相关证据定位，代价是多少？评估另行考察答案是否有证据支持、时间戳是否可用，以及证据不足时能否拒答。

- **A：纯语音文字基线。** 仅使用语音转录文本进行索引和检索；关闭画面描述、图像向量、OCR 和所有由画面衍生的特征。生成器的输入也只包含 A 获得的证据。
- **B：音频加画面。** 使用同一批视频、转录及片段配置，额外使用可核验的视觉特征或画面描述。记录具体检索、融合及生成配置。
- 固定候选范围、测试题和评价规则。若不同模型或更多计算也发生变化，应额外说明，不能把差异全部归因于画面信息。

**实现状态：A/B 是实验设计，不是已实现的命令。** 原项目的 `evaluate_retrieval.py` 中 `bm25` 和 `embedding` 路由使用片段的 `searchable_text`；该字段可能包含画面描述，不能直接宣称它们是纯语音基线。必须构造仅含转录的片段和对应索引，并核实输入确实不含视觉信息，才能运行 A。该脚本要求所有题目的 `relevant_segment_ids` 非空，不可直接将无答案题输入检索召回实验；无答案需单独测评。

## 2. 素材与隐私

1. 只使用自己录制或明确允许本用途的课程视频。分别确认个人学习、向模型服务商上传、模型训练、公开分享的许可；允许观看不等于允许公开或训练。
2. 在本机私有位置保存视频、字幕、问题和评估结果。建议使用项目已忽略的 `artifacts/` 和 `data/raw/`；删除测试素材时同时清理派生帧、音频、索引、结果和日志。
3. 不把 Canvas 链接、登录令牌、用户姓名、真实课程问题或受保护音视频提交到公共 Git。访问远程模型前确认实际会发送哪些数据及服务条款。
4. 为素材记录非敏感标识、授权来源、许可、处理方式和保留期限。公开演示优先自制素材；使用公开素材仍需逐项检查其许可及第三方内容。

可选公开素材来源：

- [MIT OCW Python 课程](https://www.ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/video_galleries/lecture-videos/)：核查[使用条款](https://www.ocw.mit.edu/pages/privacy-and-terms-of-use/)和具体素材；MIT 对 AI 训练另列署名、非商业及相同方式共享要求。
- [Yale 心理学第一讲](https://oyc.yale.edu/psychology/psyc-110/lecture-1)：包含章节及转录；遵守[使用条款](https://oyc.yale.edu/terms)，第三方内容可能不包含在许可内。

## 3. 人工题集与类型

先选择至少三个互不重叠、已获授权的视频；数量仅为方便分组的最低起点。建议试点 30–50 题并报告每类数量，不声称统计显著或覆盖广泛课程。空白采集模板见 `data/evaluation/rewindai_annotation_template.csv`，它不是原有 JSONL 评估器的直接输入。

| 类型 | 判定原则 | 最低人工证据 |
| --- | --- | --- |
| `audio` | 只听讲师即可回答 | 原声相关时间段及内容核对 |
| `visual` | 必须观看画面、图形或动作 | 原始视频帧时间、人工描述 |
| `ocr` | 必须读取画面中的文字或数字 | 原始帧时间和人工抄录 |
| `multimodal` | 必须同时利用语音和画面；任一单路都不充分 | 两类证据及无法单路回答的原因 |
| 无答案 | 对指定范围没有足够证据 | `answerable=false`，说明检查范围和原因 |

`question_type` 和 `answerable` 独立：无答案不是一种模态。每题记录视频 ID、题目、参考答案或无法回答原因、证据时间段、问题类型、是否可回答和复核状态；题目与答案都不要包含私密信息。

播放证据区间及其上下文后人工核查，标注全部实际覆盖证据的片段，并记录帧时间。无答案题不能只检查几个抽样片段后就断言整段无答案；应完整检查指定视频，或把题目范围明确限制在完整检查过的部分。对于歧义或需要视频外知识的题目，修改或排除并记录原因。

复用仓库审计格式时，阅读 [原人工复核规范](ANNOTATION_GUIDE_ZH.md)：候选题需要通过复核事件与校验后才可称为 `verified`，不能手工伪造 `review_event_id` 或仅改状态。时间需以秒记录，证据区间须属于视频，且相关 segment 与区间重叠。CSV 模板只收集草稿。可以用下方的时间诊断转换器生成单视频草稿 JSON；它不是原工具的 verified JSONL，不创建人工复核事件。正式标注仍须按原工具的格式校验和复核。

## 4. 视频级分组与冻结

- 以视频为单位划分 `development`、`validation`、`test`。同一视频的任意片段、题目和转录不得跨分组。
- 同一源视频的重新剪辑、重上传或近重复内容，以及同义改写题，先归入同一关联组并整体划分；如遇到跨视频共同内容，也应归组。
- 开发集用于实现和检查；验证集用于选择阈值和配置；测试集冻结后用于最终比较。若视频不足以划出三组，应报告限制，不能把结果称为独立测试。
- 冻结前记录问题文件、分组清单和数据版本的 SHA-256，以及代码提交、模型版本、索引参数、设备和运行时间；个人资料及敏感路径不上传。

## 5. 测量与报告

两组对同一批可回答测试题分别计算 **Recall@K**（每题前 K 个片段找回的相关片段数 ÷ 该题标记的全部相关片段数，再对题目取平均）和 **Hit@K**（前 K 个片段是否至少命中一个相关片段，再对题目取平均）。同时报告 MRR@K、K、每题或分组的数量及题目类型。Recall 的分母依赖相关片段标注是否完整；候选池之外未经检查的相关片段会导致高估。相邻片段可能同样有效，应对照视频人工核验，不应仅凭时间重叠标相关。无答案题单独报告，不放进这些正例指标。

本机比较报告还按同一批题目，将画面、融合及可选重排分别与文本检索配对比较，统计 Recall@5、Hit@5、MRR@5 的平均差值和改善、退化、持平题数。这些是描述性统计，小样本不能证明对其他课程有效。

### 本机片段相关性复核与三路比较

下面的流程用于**开发集或验证集**。先用本机时间标注页保存问题，然后启动片段相关性复核页；它按视频时间显示所有字幕片段，不展示模型排名。对照录像选择直接支持答案的全部片段。只有实际逐段核对后才声明已核验；只有完整检查整段视频并找齐所有相关片段后才声明完整。保存的是本机草稿，声明不构成独立验证；修改视频、字幕、题目或片段切分后须重新标注。

```bash
python scripts/review_retrieval_relevance.py --video PATH_TO_LOCAL_VIDEO --subtitles PATH_TO_LOCAL_SRT --labels PATH_TO_LOCAL_LABELS_CSV --video-id VIDEO_ID --split validation --output-dir artifacts/evaluation/relevance
```

打开 `http://127.0.0.1:5098/` 保存并可恢复进度。标注、问题、字幕、录像和指纹均须留在本机。无答案题不在此页的正例相关片段评估中，必须另行测量。

确认所有可回答题的相关片段和完整性均已对照录像核对后，使用刚保存的 JSON 运行：

```bash
python scripts/evaluate_local_segment_retrieval.py --video PATH_TO_LOCAL_VIDEO --subtitles PATH_TO_LOCAL_SRT --labels PATH_TO_LOCAL_LABELS_CSV --gold PATH_TO_RELEVANCE_JSON --video-id VIDEO_ID --split validation --semantic-model PATH_TO_LOCAL_QWEN3_GGUF --visual-model PATH_TO_LOCAL_CLIP_DIRECTORY --output artifacts/evaluation/local-segment-comparison.json --confirm-reviewed
```

比较只在 CPU 本机运行，对同一批题目分别获取字幕语义与画面前 5 个候选，再以 RRF（k=60）融合，按题型报告 Recall@1/5、Hit@1/5 和 MRR@1/5。脚本要求本地 Qwen3-Embedding-0.6B GGUF 文件并记录其指纹，但不会独立证明模型身份；Qwen3 文档输入截断到 512 tokens，CLIP 查询可能截断到 77 tokens。画面最多采样 100 个片段中点，未采样片段无法被画面路线找回。此实验不应用相似度阈值，与工作台默认前 3 条、带阈值的问答链路不同；它不评估答案正确率。

如需比较独立重排器，可追加 `--reranker-model 本地Qwen3-Reranker目录`。需事先检查模型许可、下载模型并安装兼容的 PyTorch、Transformers 和 Accelerate；脚本只读本机模型并使用 CPU。它对最多 20 个字幕语义候选按每批 4 条、最长 1024 tokens 重排，再取前 5 条，与原始字幕排序比较。两者均使用字幕文字，但重排路线考察更深的候选池，因此 Recall@5 才可能变化。模型文件会记录指纹，但不会独立证明其身份；CPU 运行可能较慢、占用较多内存。

报告记录输入、模型、代码指纹及部分依赖版本，但不能证明标注正确或完全复现运行。输出仅限 `artifacts/` 或 `local_data/`，不得上传课程派生结果。

#### 声明并冻结最终测试

先在其他视频上选定配置，再独立复核 `test` 视频的问题及全部相关片段，不先查看测试排名。使用下面命令冻结输入和配置；对调参使用过的**每个**开发／验证相关性文件重复提供 `--other-gold`：

```bash
python scripts/freeze_local_segment_test.py --video PATH_TO_TEST_VIDEO --subtitles PATH_TO_TEST_SRT --labels PATH_TO_TEST_CSV --gold PATH_TO_TEST_RELEVANCE_JSON --video-id TEST_VIDEO_ID --semantic-model PATH_TO_QWEN3_GGUF --visual-model PATH_TO_CLIP_DIRECTORY --other-gold PATH_TO_DEVELOPMENT_RELEVANCE_JSON --other-gold PATH_TO_VALIDATION_RELEVANCE_JSON --output artifacts/evaluation/frozen-segment-test.json --confirm-reviewed --confirm-test-isolation
```

如使用重排器，冻结和评测时都要提供相同的 `--reranker-model`。测试时在上方比较命令中改用测试素材、`--split test`，并追加相同的 `--other-gold` 文件和 `--frozen-test-manifest artifacts/evaluation/frozen-segment-test.json`。程序会在评分前后检查输入、切分、模型文件、部分代码、配置和依赖版本，并比较**已提供文件中记录的**视频哈希。它不能证明已提供全部调参数据，也不能识别近重复录像、内容重叠、错误标签、非独立复核、重复测试或中途改回的文件；声明和清单并非防篡改证明。不要使用测试结果继续调参，并在报告中保留这些限制。

答案质量需人工逐条判断：回答是否正确、引用片段是否支持结论、是否遗漏重要条件。单独报告无答案题的拒答数量和分母，同时检查有答案题的错误拒答。记录视频预处理时长、问答延迟和硬件条件；量力而行，不推断未测的价格或显存。

本地回答复核汇总还报告人工判定有证据、无证据的回答数及其在非拒答回答中的支持比例；全部拒答时比例未定义。这一比例不是检索 Precision，也不等同于答案正确率：有证据的回答仍可能没有回答问题。

新生成的回答复核草稿会保留问题类型，并在汇总中按已标注类型报告题数与指标；未记录类型的旧草稿归入 `unlabeled`，不会自动推断。小样本分组仅作描述，题目被标为视觉类型也不代表系统已理解画面。

汇总还会统计已记录的本地问答耗时中位数和最近秩 P95，并显示记录数及缺失数。计时包括逐题检索和回答生成，不含模型加载、索引构建与人工复核；各题顺序执行，缓存和机器负载会影响结果。小样本 P95 不稳定，不能当作普遍性能保证。

如果某题回答生成失败，但重新检索成功，草稿会保留失败状态并继续处理后续问题。页面显示的是失败后重新检索的候选，不能当作失败时模型实际收到的内容。生成失败不计为拒答或错误，评分前必须修正原因并重新生成草稿；已保存的复核进度绑定原草稿，不会自动转移到新草稿。如果重新检索也失败，整批仍会停止且不会保存草稿。

对于开发集或验证集的新草稿，可以点击“建议复用未变判断”，从最近的本机进度或复核副本中查找可复用判断。系统检查已记录的输入指纹、配置、问题、参考、回答、证据与模型实际收到的文字；任何变化、生成失败、缺少来源记录或最终测试集均不复用。耗时允许不同。已有选择不会被覆盖，建议仍需逐题人工核对和确认；这些检查不能独立证明原判断正确。最多检查最近 100 个符合名称的本机文件。

实验记录模板：`data/evaluation/rewindai_experiment_template.md`。分别记录每组配置、分母、结果、错误例子与限制，禁止将候选题、模拟演示或未运行实验标为正式结果。必要时先在私有目录整理题目，再使用 `scripts/validate_questions.py --help` 和 `scripts/evaluate_retrieval.py --help` 检查工具要求；检索工具需要真实索引和模型，不能只靠模板运行。
# 本地时间范围诊断

## 使用本地页面收集题目草稿

安装 PyAV 后，可为单个视频启动标注页面（先选定开发、验证或测试划分，每个视频使用独立的新输出文件）：

```powershell
python scripts/temporal_label_server.py --video PATH_TO_LOCAL_VIDEO --subtitles PATH_TO_LOCAL_SRT --video-id lecture_dev --split development --output artifacts/evaluation/lecture-dev-drafts.csv
```

在同一台电脑打开 `http://127.0.0.1:5095/`。播放视频并检查上下文，填写问题和参考答案，用按钮记录当前播放时间作为证据起止；无答案题要求声明已针对该问题检查完整视频。页面无法验证用户是否观看或独立复核；这些题仍须人工核验，不能因为保存而称为正式标签。

提供字幕后，开发集和验证集可在页面运行本地关键词检索诊断；测试集在此页面不可运行，以便保留给后续独立评估。诊断要求首条结果覆盖参考时间范围至少 50%，并报告时间证据 Precision、Recall、F1：有答案题返回错误时间时同时记为假阳性和假阴性。这些指标不代表回答正确率，标签质量和字幕覆盖范围都需核对。运行前后会计算视频和字幕的指纹，但这不能证明标注时使用的是同一份字幕。

页面也会统计有多少可回答问题无法由任何单个已索引片段达到 50% 覆盖要求，并在非零时显示提示。该检查只比较时间区间：它不能判断证据是否相关，也不会修改标签或评分。出现提示时，应结合录像核查参考范围和切分方式；较长的问题可能需要多个证据区间。

命令行的字幕诊断和关键词／语义比较也会在 `coverage_feasibility` 中报告这一统计，使用运行时指定的覆盖阈值。

诊断另外报告返回片段的时间并集是否覆盖了指定比例的参考区间，重叠部分只计算一次；此项不改变首片段评分。它仍只是时间覆盖，不证明这些片段在语义上构成完整证据，也不能替代人工检查录像。

安装 `[qwen-semantic]` 依赖并在启动命令追加 `--semantic-model PATH_TO_QWEN3_GGUF`，页面会对草稿运行单模型语义诊断。模型文件需预先保存在本地，使用 CPU 和固定的、未经校准的 0.4 相似度阈值。诊断仅检索字幕；小样本和未经核验标签不能支持普遍性的准确率结论。运行前后会检查模型文件指纹。如果本地模型不可用，去掉该参数可运行关键词诊断。

诊断前勾选“本地查看逐题时间”，页面会显示每题的参考区间、两种方法返回的时间及跳转按钮，便于回看错误。时间匹配不证明片段能够回答问题。逐题信息及结果可能包含私人课程内容，保留在本机，避免分享截图；页面不会自动导出结果。

如需保存汇总，运行前勾选“保存本地汇总”。服务会在标注 CSV 同目录的 `reports/` 下创建新 JSON，不覆盖旧报告；报告包含汇总计数、配置和输入指纹，不包含逐题文本或逐题结果。它仍是未经核验的草稿，指纹和汇总也可能是敏感数据，必须留在本机。写入前会重新检查输入指纹和标注版本。

仅在开发集，页面可按字幕片段顺序抽取最多 12 段，供浏览并跳转原视频。这些只是起点，不是自动生成的问题或核验过的答案；字幕可能有误，根据字幕选题也可能使开发集产生偏差。验证集和测试集不提供该功能，选题仍需对照原视频。

完成开发和验证后，可使用[画面搜索实验指南](LOCAL_VISUAL_EXPERIMENT.md#frozen-test-after-development)中单独的冻结清单流程评估按视频留出的测试 CSV。该流程要求人工声明已核对题目，并检查数据、模型、部分代码和参数；声明不能证明评审独立，也无法自动识别近重复视频。不要用测试结果继续调整配置，课程标签、指纹和结果仍只保存在本机。

如需评估最终**回答**而不只是时间定位，另见[本地回答实验的冻结测试流程](LOCAL_ANSWER_EXPERIMENT.md#frozen-answer-test-after-development)。先固定模型和参数、核对测试标签，再由人工逐题检查生成回答与证据后计算 Precision、Recall 和 F1；这些指标仍取决于标签与评审质量，不能自动证明独立性。

每个 CSV 同时只运行一个标注服务。页面可以添加和修改自身创建的草稿，但不能删除记录；带有额外复核数据的记录需停止服务后在外部编辑。外部编辑 CSV 前都要停止服务。输出 CSV 及旁边的 `.source.json` 含私人数据，应留在本机。启动时用视频 SHA-256 绑定输出；运行期间通过文件大小和修改时间检测变化，不能证明内容始终未变。

对已经准备好的本地视频、字幕与问题 JSON，可运行：

```powershell
python scripts/diagnose_transcript.py --video "C:\path\video.mp4" --subtitles "C:\path\video.srt" --questions "C:\path\questions.json"
```

JSON 包含 `questions` 数组；每条至少有 `question`。有答案的问题还需 `evidence_start_seconds` 和 `evidence_end_seconds`；无答案问题设置 `expected_answerable` 为 `false`。程序报告返回片段是否与参考时间有正时长交集、单个片段覆盖参考时间的比例，以及无答案问题是否拒答。时间重叠不等于回答正确，比例也不是模型准确率；只有先对问题及证据独立人工复核，诊断才有意义。输出可能包含问题文本，私人资料请留在本地。


时间证据的 F1 按 `2TP / (2TP + FP + FN)` 计算：存在漏检但没有预测时，Precision 未定义而 F1 为 0；没有预测且没有正例时 F1 未定义。该指标只衡量时间范围覆盖，不代表回答正确。

诊断输出还单独报告可回答问题中首条及任一返回片段达到覆盖阈值的数量、可回答问题的拒答数，以及无答案问题的拒答和返回候选数，各自包含分母和比例。任一返回片段的命中率只覆盖系统实际返回的候选，不等于标注了所有相关片段的标准 Recall@K。可选 `question_type` 字段用于分组；未标类型的题归为 `unlabeled`。零分母返回 `null`。输出包含问题和时间范围，私人课程评估必须留在本机。

`compare_time_retrieval.py` 的关键词与语义对照报告也按题型分别列出时间证据 Precision、Recall、F1、命中和拒答计数。每组的分母可能很小；无正例或无预测时部分比率为 `null`，不应把它们解释为零或总体表现。两组都是字幕文本检索，即使问题标为 `visual`，也不能据此宣称已经比较视觉检索。`--aggregate-only` 不含逐题结果，但题型名称、计数和可选哈希仍可能敏感，私人评估报告仍须留在本机。

对照报告还会记录每种方法逐题查询的墙上时钟耗时汇总（均值、中位数和最近秩法 P95，单位毫秒）；单独诊断也给出逐题耗时。计时从调用检索开始到结果转换完成，包括首题，但不包括模型加载、视频转录、索引构建、浏览器交互或网络。对照先运行全部关键词问题，再运行全部语义问题，没有预热或随机交错；缓存和系统负载会影响结果。小样本的 P95 不稳定，不能把它当作用户实际等待时间或通用性能保证。

## 将本地 CSV 草稿转换为时间诊断输入

使用 `data/evaluation/rewindai_annotation_template.csv` 的列填写本地 CSV；每题需要唯一 ID、视频 ID、题型、`true`/`false` 是否可回答。可回答题填写秒数范围；无答案题留空时间并填写实际完整检查过的 `checked_scope`。按单个视频和可选 split 转换：

```powershell
python scripts/prepare_temporal_questions.py --csv PATH_TO_LOCAL_CSV --video-id VIDEO_ID --split test --output artifacts/evaluation/local-draft.json
python scripts/diagnose_transcript.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --questions artifacts/evaluation/local-draft.json
```

输出仅允许写入 `artifacts/` 或 `local_data/` 下的新 JSON 文件，不覆盖旧文件。转换器保留题目但始终标为 `csv_import_unverified_draft`，不核实视频 ID 与文件是否匹配，也不验证观看行为、证据内容或完整检查范围；调用者须自行核对。它不生成原复核工具所需的事件与 verified JSONL。题目、答案、视频、字幕和结果可能含私人课程资料，均应留在本机。

为减少题目与素材配错，可在转换命令中同时传入 `--video PATH_TO_VIDEO --subtitles PATH_TO_SRT`。转换器在本机计算两个文件的 SHA-256 并写入草稿；时间诊断、语义诊断及两者对照会在运行前后检查指纹，不匹配时停止。未绑定的旧题集仍可运行，输出 `source_binding: unbound`，须人工核对。指纹绑定不证明题目、答案或时间戳正确；计算长视频哈希可能耗时，指纹及题目也应随私人资料保留在本机。

### 检查视频划分

转换 CSV 时，同一视频的所有行必须属于同一个 split，不能混用未分配和已分配行；未填任何 split 的题目只能作为未分组草稿。转换并绑定多个本地题目文件后，可以执行：

```powershell
python scripts/audit_temporal_splits.py --questions artifacts/evaluation/dev.json --questions artifacts/evaluation/test.json
```

工具检查跨 split 的相同视频 ID、已记录的视频 SHA-256 以及忽略大小写和空白差异的完全相同问题，只输出汇总数量。缺少指纹或不足两个 split 时结果为不确定；发现冲突或不确定时命令返回非零状态。它读取保存的指纹元数据，不会重新验证实际视频内容；诊断运行时才会验证绑定。它不能识别视频剪辑、重编码、同一课程相邻录像或问题改写，必须另行分组和人工检查。通过检查也不证明测试集独立或标签正确。


## 复核记录转为时间证据评估

完成实际人工复核、导出并冻结视频级划分后，可以把其中一个视频和 split 的题目转换为本地时间诊断输入：

```powershell
python scripts/reviewed_to_temporal.py --questions PATH_TO_FROZEN_JSONL --manifest PATH_TO_SPLIT_MANIFEST --candidates PATH_TO_CANDIDATES_JSONL --events PATH_TO_REVIEW_EVENTS_JSONL --segments PATH_TO_SEGMENTS_JSONL --video PATH_TO_LOCAL_VIDEO --subtitles PATH_TO_LOCAL_SRT --video-id VIDEO_ID --split test --output artifacts/evaluation/reviewed-temporal.json
python scripts/diagnose_transcript.py --video PATH_TO_LOCAL_VIDEO --subtitles PATH_TO_LOCAL_SRT --questions artifacts/evaluation/reviewed-temporal.json
```

转换时检查冻结文件与清单的 SHA-256、完整复核事件链、每题最新接受事件及其内容、split 一致性、视频 ID 和片段记录中的源文件路径；明确标为模拟的 smoke 复核会被拒绝。转换输出绑定当前视频和字幕哈希，且只能写入本地忽略目录中的新文件。

这些都是一致性检查：本地事件和清单不是防篡改证明，无法证明人工复核独立或内容正确；片段记录的源路径也无法证明文件内容在复核时与转换时相同。转换所用字幕可以与复核时的字幕不同，因此需单独记录并核对版本。脚本不会把输出自动标记为正式指标，仍须人工核验和确保测试集不参与调参。私人录像、问题、哈希、日志与结果均保留在本机。
