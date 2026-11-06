# 公开素材：AI 医疗伦理短视频

此视频用于本地字幕检索和播放器测试，不是项目自制内容，也不代表模型已经经过真实准确率评估。

- 标题：Ethics of AI in Digital Medicine – Explanation why transparent explainable AI is important
- 作者 / 来源：KI-Campus，通过 [Wikimedia Commons 素材页](https://commons.wikimedia.org/wiki/File:Ethics_of_AI_in_Digital_Medicine_%E2%80%93_Explanation_why_transparent_explainable_AI_is_important.webm) 发布。
- 许可：[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。使用时请核对素材页面的具体条件，保留署名、许可链接，并标注修改；衍生作品可能需要相同方式共享。
- 视频约 97 秒，带英语音频；可在素材页面找到原文件。
- 英文字幕可见 [Wikimedia TimedText 页面](https://commons.wikimedia.org/wiki/TimedText:Ethics_of_AI_in_Digital_Medicine_%E2%80%93_Explanation_why_transparent_explainable_AI_is_important.webm.en.srt)。

将合法取得的视频和 SRT 保存到本机后，可以运行：

```bash
python scripts/run_transcript.py --video /path/to/ai_ethics_kicampus.webm --subtitles /path/to/ai_ethics_kicampus.en.srt
```

字幕来自配套页面，但仍需观看视频确认字幕与画面。视频和字幕均保存在本地，不纳入此仓库。此模式使用关键词与原文摘录，不分析视频画面，也不是医疗建议。

## 第二个公开学习视频

- [Artificial intelligence explained in 2 minutes](https://commons.wikimedia.org/wiki/File:Artificial_intelligence_explained_in_2_minutes_-_What_exactly_is_AI%3F.webm)，作者 KI-Campus，时长约 2 分 23 秒。
- 视频许可：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。请保留作者署名、许可链接及修改说明，分享改编材料时遵循相同方式共享条件。
- [英文字幕页面](https://commons.wikimedia.org/wiki/TimedText:Artificial_intelligence_explained_in_2_minutes_-_What_exactly_is_AI%3F.webm.en.srt)。视频和字幕保留在本地，不纳入仓库。


## 本地真实模型工作台流程检查

2026 年 11 月 6 日，使用本页面所述公开教学视频及其英文字幕，在隔离的本地数据库中启动 Qwen3-Embedding-0.6B 和 Qwen2.5-3B-Instruct 回答模式，均在 CPU 推理。输入一条字幕相关问题后，页面显示生成回答及三个带时间戳的字幕候选；选择其中一个 0:19 片段时页面显示该起点，另一个片段可加入本地复习清单。此次未使用私人课程资料，也未将资料上传。

这是一次功能冒烟测试，不是答案正确率或视频理解的评测；字幕、回答和参考问题仍需对照源视频独立核验。
