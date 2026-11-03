# RewindAI · 回课 AI

**Find it. Verify it. Learn it.**  
**找到知识，回看证据，学得更透。**

RewindAI is an AI-powered learning workspace for finding and reviewing moments in lecture videos. Ask a question, inspect timestamped evidence, jump to the video, and save a clip with notes.

回课 AI 是一个课程视频学习工作台：针对视频提问、查看有时间戳的证据、跳转到原视频，并收藏需要复习的片段。

> **Project status / 项目状态**：本项目处于开发和验证阶段。CPU 演示使用预设文本与简化检索，不代表真实视频识别或模型效果；完整多模态流程需独立安装模型、准备视频并完成评测。仅适合本地使用，不要直接部署到公网。

## What it does · 当前功能

- 查看从已加载索引中识别的视频，创建课程并整理视频。
- 对视频资料提问，查看答案和证据；证据包含时间戳，可跳转到可用的视频文件。
- 保存或删除片段收藏，为收藏编辑复习笔记。
- 对回答和证据提交反馈，供后续分析。
- 区分演示模式和完整模式，避免把占位演示当作真实模型结果。
- 提供本地字幕检索模式：读取自备视频和 SRT/VTT 字幕，按时间戳查找和播放片段，无需模型或联网。
- 支持语音、关键帧、混合检索、重排和人工评测，完整多模态模型流程仍需独立验证。

**当前限制：**网页不自动下载 Canvas 录像。在字幕模式下可输入本机视频和字幕路径来加载视频；视频路径、字幕副本、课程归属、收藏和笔记存储在本地数据库。重启时会恢复未被移动或修改的视频；如果视频丢失或改变，界面会提示恢复原文件。字幕副本不会自动随外部字幕文件更新。完整多模态模式需要预处理并建立索引。演示包未包含真实示例视频，播放器可能显示不可用。问答默认搜索全部索引，也可选择仅搜索当前视频；单视频搜索需要扫描完整索引，在大型数据集上可能较慢。尚不支持按整个课程检索。服务仅接受本机 Host 和同源浏览器请求；没有账号、多用户隔离或生产级鉴权，远程使用需要本机 SSH 转发，不支持通过非本机域名直接访问。

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

如果你有本地视频及其配套的 UTF-8 `.srt` 或 `.vtt` 字幕，可以在项目环境中运行：

```bash
python scripts/run_transcript.py --video /path/to/lecture.mp4 --subtitles /path/to/lecture.srt
```

此模式无需下载模型、无需联网，适合资源有限的电脑。它只进行**关键词匹配和字幕原文摘录**，不是语音识别、AI 生成或视频画面理解；不会自行生成字幕，也不能保证跨语言语义检索。只能处理外置字幕文件。如果字幕不足以回答问题，应该回看视频核对。单独运行时默认打开 `http://127.0.0.1:5000`。

要同时学习多个本地视频，可以按相同顺序重复指定视频和字幕：

```bash
python scripts/run_transcript.py --video lecture1.mp4 --video lecture2.mp4 --subtitles lecture1.srt --subtitles lecture2.srt
```

也可以省略所有 `--subtitles`，程序会为每个视频寻找同名的 `.srt` 或 `.vtt` 文件。文件只在本机读取，加载后可以在网页中将视频归入课程并选择单视频搜索。

若视频没有字幕，可以另行安装 faster-whisper 并使用 [本地转字幕指南](docs/LOCAL_TRANSCRIPTION.md)。默认使用本地缓存和 CPU，不向云端上传视频或音频；首次获取模型需显式允许下载。转录脚本有模拟测试，但真实模型转录尚未验证。

### 使用完整多模态模型

完整流程涉及视频预处理、构建索引和启动真实服务。推荐 Ubuntu 22.04、Python 3.11+、FFmpeg、24 GB 以上 NVIDIA 显存、64 GB 内存与 100 GB 磁盘；具体资源取决于视频和模型配置。请先阅读 [本地视频导入指南](docs/LOCAL_VIDEO_IMPORT.md) 和 `config.toml`，再根据自己的设备验证。模型权重和依赖的首次获取可能需要联网。

安装好模型依赖、FFmpeg 并准备好本地缓存后，可处理已有授权的视频文件或文件夹：

```bash
python scripts/ingest_local.py /path/to/authorized-video.mp4 --language en
```

该命令依次调用预处理和索引脚本，将结果保存到新的 `artifacts/ingestions/` 子目录，不复制原视频。默认启用模型中心离线模式；模型不在本地时需要显式允许下载。命令输出真实服务启动示例。此流程尚未在真实模型与课程视频上端到端验证。

本项目区分**本地索引**和**模型训练**：提取片段并建立检索索引通常不会改变模型权重。使用任何课程视频或个人资料前，请确认你有权使用。课程录制、转录、索引、收藏和反馈默认应保留在本地，不要上传到公开仓库；若计划使用云模型，须先确认数据会发往何处以及相应授权。

## Architecture · 系统流程

```text
Video → speech transcript + keyframes / visual descriptions
      → timestamped segments → hybrid retrieval → reranking
Question ──────────────────────────────────────────┘
      → evidence-grounded answer + timestamps → review and bookmarks
```

CPU 演示仅使用模拟片段、词汇检索和简单回答拼接；以上完整模型流程不代表已经验证过模型质量。

## Evaluation · 评估

上游候选问题集用于诊断，不是经过完整人工核验的正式金标准。不要把历史候选集中的结果写成经验证的准确率。课程视频测试应逐条核验答案和证据，按视频划分数据，分别检查语音、画面、综合和无答案问题。

- [中文评估方案](docs/REWINDAI_EVALUATION_ZH.md)
- [初步实际评估与可复现方法](docs/INITIAL_EVALUATION.md)
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

