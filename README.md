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
- 在已有 VideoRAG 流水线上保留语音、关键帧、混合检索、重排和人工评测能力，完整模型运行仍需独立验证。

**当前限制：**网页并不自动下载 Canvas 录像，也不直接完成上传、处理及索引；视频需要先通过本地处理流程导入。演示包未包含真实示例视频，播放器可能显示不可用。当前问答基于已加载的全部索引，不保证只检索某一课程。服务仅接受本机 Host 和同源浏览器请求；没有账号、多用户隔离或生产级鉴权，远程使用需要本机 SSH 转发，不支持通过非本机域名直接访问。

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

测试与代码检查：

```bash
python -m pytest -q
python -m ruff check src/video_rag/api scripts/ingest_local.py tests/test_ingest_local.py tests/test_library_api.py
```

## Real video workflow · 真实视频流程

完整流程涉及视频预处理、构建索引和启动真实服务。原项目推荐 Ubuntu 22.04、Python 3.11+、FFmpeg、24 GB 以上 NVIDIA 显存、64 GB 内存与 100 GB 磁盘；具体资源取决于视频和模型配置。请先阅读 [本地视频导入指南](docs/LOCAL_VIDEO_IMPORT.md)、[原始 VideoRAG 技术说明](docs/UPSTREAM_VIDEORAG.md) 和 `config.toml`，再根据自己的设备验证。模型权重和依赖的首次获取可能需要联网。

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

CPU 演示仅使用模拟片段、词汇检索和简单回答拼接；以上完整模型流程是原项目描述的方案，不代表本次已经验证过模型质量。

## Evaluation · 评估

上游候选问题集用于诊断，不是经过完整人工核验的正式金标准。不要把历史候选集中的结果写成经验证的准确率。课程视频测试应逐条核验答案和证据，按视频划分数据，分别检查语音、画面、综合和无答案问题。

- [中文评估方案](docs/REWINDAI_EVALUATION_ZH.md)
- [人工标注模板](data/evaluation/rewindai_annotation_template.csv)
- [实验记录模板](data/evaluation/rewindai_experiment_template.md)
- [完整产品和研发计划](docs/REWINDAI_PLAN_ZH.md)

## Attribution · 来源与贡献

RewindAI builds on [VideoRAG](https://github.com/zhuzhenxiang93-create/VideoRAG). The upstream project supplies the existing video RAG pipeline, evaluation tools and initial UI. RewindAI adds a learning workspace and product workflow; see the source and commit history for the precise changes. 原作者的技术说明保留在 [docs/UPSTREAM_VIDEORAG.md](docs/UPSTREAM_VIDEORAG.md)。

The repository owner has confirmed that they authored the supplied base project or have permission to modify and republish it. No specific open-source licence has been selected in this repository yet; consult the owner before reuse. Upstream model weights, datasets, videos and third-party dependencies have their own terms.

## Links · 项目链接

- RewindAI: <https://github.com/laizhen1/RewindAI>
- VideoRAG: <https://github.com/zhuzhenxiang93-create/VideoRAG>
- Whisper: <https://github.com/openai/whisper>
- FAISS: <https://github.com/facebookresearch/faiss>
