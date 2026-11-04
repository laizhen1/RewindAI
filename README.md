# RewindAI · 回课 AI

**Find it. Verify it. Learn it.**  
**找到知识，回看证据，学得更透。**

RewindAI is an AI-powered learning workspace for finding and reviewing moments in lecture videos. Ask a question, inspect timestamped evidence, jump to the video, and save a clip with notes.

回课 AI 是一个课程视频学习工作台：针对视频提问、查看有时间戳的证据、跳转到原视频，并收藏需要复习的片段。

> **Project status / 项目状态**：本项目处于开发和验证阶段。CPU 演示使用预设文本与简化检索，不代表真实视频识别或模型效果；完整多模态流程需独立安装模型、准备视频并完成评测。仅适合本地使用，不要直接部署到公网。

## What it does · 当前功能

- 查看从已加载索引中识别的视频，创建、重命名或删除课程并整理视频；删除课程会保留视频和笔记。
- 对视频资料提问，查看答案和证据；证据包含时间戳，可跳转到可用的视频文件。字幕模式可按需展开同一视频的前后字幕，作为未验证的上下文；安装 PyAV 后也可查看片段中点附近的一帧画面，画面仅供人工核验。
- 保存或删除片段收藏，为收藏编辑复习笔记。
- 分别对回答和证据提交有用或需改进的反馈，并可填写原因；反馈保存在本地数据库。
- 区分演示模式和完整模式，避免把占位演示当作真实模型结果。
- 提供本地字幕检索模式：读取自备视频和 SRT/VTT 字幕，按时间戳查找和播放片段，无需模型或联网。
- 支持语音、关键帧、混合检索、重排和人工评测，完整多模态模型流程仍需独立验证。

**当前限制：**网页不自动下载 Canvas 录像。在字幕模式下可输入本机视频和字幕路径来加载视频；视频路径、字幕副本、课程归属、收藏和笔记存储在本地数据库。重启时会恢复未被移动或修改的视频；如果视频丢失或改变，界面会提示恢复原文件，或移除失效的本地索引后重新导入；原视频和收藏笔记会保留，旧版片段收藏可能无法回看，需要重新收藏。视频变化通过文件大小和修改时间检测，并非内容完整性校验。字幕副本不会自动随外部字幕文件更新；字幕模式可先选择视频并移除其本地索引，再重新加载更新后的字幕。原视频、原字幕和收藏备注不会删除；来自旧字幕的收藏可能无法回看。若使用命令行显式指定视频，重新运行相同启动命令会再次加载该视频。完整多模态模式需要预处理并建立索引。演示包未包含真实示例视频，播放器可能显示不可用。问答默认搜索全部索引，也可选择当前视频或当前视频所属课程；指定范围的搜索需要扫描完整索引，在大型数据集上可能较慢。服务仅接受本机 Host 和同源浏览器请求；没有账号、多用户隔离或生产级鉴权，远程使用需要本机 SSH 转发，不支持通过非本机域名直接访问。

## Quick start · 快速开始

运行 CPU 演示无需下载大模型，适合检查界面和交互：

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_demo.py
```

在浏览器打开 `http://127.0.0.1:5000`。演示数据是两条预设片段，并非读取真实课堂视频。首次启动无真实视频文件时，视频播放和时间跳转无法展示实际画面。

在 Windows 上完成环境安装后，也可以双击 `Start-RewindAI-Demo.cmd` 启动演示，然后在浏览器打开上面的本机地址。关闭命令窗口即可停止服务。

测试与代码检查：

```bash
python -m pytest -q
python -m ruff check src/video_rag/api scripts/ingest_local.py tests/test_ingest_local.py tests/test_library_api.py
```

## Real video workflow · 真实视频流程

### 使用已有字幕（低配置）

在 Windows 安装好环境后，双击 `Start-RewindAI-Workspace.cmd`，然后打开 `http://127.0.0.1:5000`；也可以运行 `python scripts/run_transcript.py`。无需在启动命令中指定视频：在网页左侧输入本机视频与字幕路径即可加载。之前通过网页添加且文件未发生改变的视频会从本地数据库恢复。还没有加载视频时，先添加视频再搜索。关闭启动窗口即可停止服务。

如果你有本地视频及其配套的 UTF-8 `.srt` 或 `.vtt` 字幕，可以在项目环境中运行：

```bash
python scripts/run_transcript.py --video /path/to/lecture.mp4 --subtitles /path/to/lecture.srt
```

要在字幕模式下查看命中片段的画面，可安装 `python -m pip install -e ".[asr]"`（包含 PyAV）；画面在本机按需解码，不上传，也不代表系统理解画面。

此模式无需下载模型、无需联网，适合资源有限的电脑。它只进行**关键词匹配和字幕原文摘录**，不是语音识别、AI 生成或视频画面理解；不会自行生成字幕，也不能保证跨语言语义检索。只能处理外置字幕文件。如果字幕不足以回答问题，应该回看视频核对。单独运行时默认打开 `http://127.0.0.1:5000`。

要同时学习多个本地视频，可以按相同顺序重复指定视频和字幕：

```bash
python scripts/run_transcript.py --video lecture1.mp4 --video lecture2.mp4 --subtitles lecture1.srt --subtitles lecture2.srt
```

也可以省略所有 `--subtitles`，程序会为每个视频寻找同名的 `.srt` 或 `.vtt` 文件。文件只在本机读取，加载后可以在网页中将视频归入课程并选择单视频搜索。

若视频没有字幕，可以另行安装 faster-whisper 并使用 [本地转字幕指南](docs/LOCAL_TRANSCRIPTION.md)。Windows 用户完成安装并缓存模型后，可双击 `Start-RewindAI-Transcribe.cmd` 选择视频进行本地转录，再打开搜索界面；新视频会保存到本地资料库，以便下次恢复。已保存的视频不会被自动覆盖。默认使用本地缓存和 CPU，不向云端上传视频或音频；首次获取模型需显式允许下载。转录已在两段公开短视频上使用本地模型验证；课程录制的效果仍需评估。

### 实验性英语语义搜索

可选择本地 ONNX 英语向量模型尝试搜索相近表达。先安装 `python -m pip install -e ".[semantic]"`，并自行准备 [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) 的 `tokenizer.json` 和 `onnx/model_quantized.onnx` 到同一模型目录，然后启动：

```powershell
python scripts/run_transcript.py --semantic-model PATH_TO_LOCAL_MODEL_DIRECTORY
```

Windows 用户也可以在准备好该模型后双击 `Start-RewindAI-Semantic.cmd`。启动器默认查找项目的 `artifacts/hf-cache` 中的固定版本 `08a308f628bc9d6774b7922f319eb1b65afa1a82`；如果模型放在其他位置，可先设置 `REWINDAI_SEMANTIC_MODEL_DIR` 指向模型目录。启动器不下载模型。

语义模式仅在本机 CPU 推理，不下载模型或上传字幕。它使用尚未校准的余弦阈值 0.4，只针对英语实验，可能返回不相关片段；它不生成回答，也不理解视频画面。索引较多字幕时启动和导入可能变慢。结果需回看核对。此前小样本评估有较多误报，详见 [评估报告](docs/INITIAL_EVALUATION.md)。不传该参数时仍使用默认关键词搜索。

### 实验性本地画面搜索

可选的 CLIP 画面检索在 CPU 上运行，按每个字幕片段中点抽取一帧，用英文描述查找相似画面。它与字幕检索分别运行，需已有视频和字幕；不生成回答，也不保证画面能回答问题。由于尚未校准无匹配阈值，**即使没有相关画面也会返回候选**，必须点击画面和原视频核对。可搜索全部视频、当前视频或当前视频所属课程，每次最多搜索 100 个片段的中点画面；范围更大时会在各视频中均匀抽样并提示可能遗漏，范围超过 100 个视频时需要缩小。首次搜索、切换范围及视频库变化后可能需要重新索引。

安装 `python -m pip install -e ".[semantic]" "av>=11,<17"`，并将 [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32) 的 `tokenizer.json` 与 `onnx/model_quantized.onnx` 下载到本机模型目录（模型文件约 154 MB，遵循模型自身许可）。然后运行：

```powershell
python scripts/run_transcript.py --visual-model PATH_TO_LOCAL_CLIP_MODEL
```

Windows 用户准备好模型后，也可双击 `Start-RewindAI-Visual.cmd`。它默认查找项目本地缓存中的固定版本 `d15189d7028b43f1d3e65039190477f6af591c2a`；如使用其他目录，先设置 `REWINDAI_VISUAL_MODEL_DIR`。启动器本身不会下载模型。

打开本机工作台并加载已授权的视频和字幕，输入英文画面描述，点击“实验性画面搜索”。也可以选择“对照字幕与画面”，一次检索并分别查看字幕候选与画面候选；它不融合分数或生成多模态回答。若画面索引失败，字幕结果仍会显示。视觉索引、查询和帧处理都在本机 CPU 进行；代码不会上传课程视频。不要把私人截图、索引、字幕或模型输入提交到 GitHub。可在本地网页逐帧标注正例与无匹配问题，再计算采样帧级指标。详细模型版本、标注流程、已知限制和公开视频验证见 [画面搜索实验](docs/LOCAL_VISUAL_EXPERIMENT.md)。

### 使用完整多模态模型

完整流程涉及视频预处理、构建索引和启动真实服务。推荐 Ubuntu 22.04、Python 3.11+、FFmpeg、24 GB 以上 NVIDIA 显存、64 GB 内存与 100 GB 磁盘；具体资源取决于视频和模型配置。请先阅读 [本地视频导入指南](docs/LOCAL_VIDEO_IMPORT.md) 和 `config.toml`，再根据自己的设备验证。模型权重和依赖的首次获取可能需要联网。

安装好模型依赖、FFmpeg 并准备好本地缓存后，可处理已有授权的视频文件或文件夹：

```bash
python scripts/ingest_local.py /path/to/authorized-video.mp4 --language en
```

该命令依次调用预处理和索引脚本，将结果保存到新的 `artifacts/ingestions/` 子目录，不复制原视频。默认启用模型中心离线模式；模型不在本地时需要显式允许下载。命令输出真实服务启动示例。此流程尚未在真实模型与课程视频上端到端验证。

本项目区分**本地索引**和**模型训练**：提取片段并建立检索索引通常不会改变模型权重。使用任何课程视频或个人资料前，请确认你有权使用。课程录像、转录、索引、问题、收藏、反馈及包含课程内容的评估结果应保留在本地，不上传 GitHub、云模型或其他第三方。`.gitignore` 排除了常见媒体、字幕、数据库和 `artifacts/`、`local_data/`，但不能识别散落在其他位置或改名后的所有私人内容；发布前仍须检查。

## Architecture · 系统流程

```text
Video → speech transcript + keyframes / visual descriptions
      → timestamped segments → hybrid retrieval → reranking
Question ──────────────────────────────────────────┘
      → evidence-grounded answer + timestamps → review and bookmarks
```

CPU 演示仅使用模拟片段、词汇检索和简单回答拼接；以上完整模型流程不代表已经验证过模型质量。

工作台复习清单中的“本地反馈记录”可以查看和删除最近 100 条反馈；本机服务运行时也可通过 `http://127.0.0.1:5000/api/feedback` 查看。记录包含问题、评价、原因、可选的证据片段 ID 与反馈来源（字幕、画面或整体结果）；旧版反馈保留为来源未知。页面按来源统计全部本地反馈，记录列表则只显示最近 100 条。反馈由用户自愿提交且可能重复，数量不是准确率、Precision、Recall 或 F1；评估模型仍需独立标注的测试集。反馈可能包含私人课程内容，不应公开或上传。

## Evaluation · 评估

上游候选问题集用于诊断，不是经过完整人工核验的正式金标准。不要把历史候选集中的结果写成经验证的准确率。课程视频测试应逐条核验答案和证据，按视频划分数据，分别检查语音、画面、综合和无答案问题。

**可复现的初步结果：**在 TutorialVQA 的 18 个教学视频、294 个问题上，按参考字幕句子范围计算，关键词检索的 Precision / Recall / F1 为 **0.394 / 0.095 / 0.153**；实验性英语语义检索为 **0.388 / 0.129 / 0.194**。这是字幕句子重合度，不是视频时间戳或回答正确率；数据集不含无答案问题，且此前整体数据已用于探索，不能视为完全未接触过的最终测试集。

画面检索仅完成公开短视频的五帧初步诊断；标签由助手拟定且未经独立核验，尚不能提供可靠的视觉 F1。课程录像的隐私数据、标注和评估细节保留在本机。方法、计数和限制见下列文档。

- [中文评估方案](docs/REWINDAI_EVALUATION_ZH.md)
- [初步实际评估、独立 TutorialVQA 字幕检索测试与可复现方法](docs/INITIAL_EVALUATION.md)
- [人工标注模板](data/evaluation/rewindai_annotation_template.csv)
- [实验记录模板](data/evaluation/rewindai_experiment_template.md)
- [完整产品和研发计划](docs/REWINDAI_PLAN_ZH.md)
- [可用于本地演示的公开教育视频及许可](docs/PUBLIC_SAMPLE_ZH.md)

## Licence · 许可证

RewindAI's code is available for non-commercial use, modification and redistribution under the [RewindAI Non-Commercial Use License](LICENSE). Commercial use requires separate written permission. Model weights, datasets, videos and third-party dependencies have their own terms.

本项目代码允许非商业用途的使用、修改和再分发，详见 [许可证](LICENSE)；商业用途需要另行取得书面许可。模型、数据集、视频和第三方依赖遵循各自的许可条款。

## Links · 项目链接

- RewindAI: <https://github.com/laizhen1/RewindAI>
- Whisper: <https://github.com/openai/whisper>
- FAISS: <https://github.com/facebookresearch/faiss>
