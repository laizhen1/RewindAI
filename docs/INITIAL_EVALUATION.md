# Initial local retrieval diagnostic

This is a **small provisional engineering diagnostic**, not a validated benchmark or a measure of answer accuracy. The video and reference subtitles are public; the nine questions in [`public_sample_questions.pending.json`](../data/evaluation/public_sample_questions.pending.json) are draft labels, have not been independently verified against the video, and were designed while developing the system. They are not a held-out test set.

## Setup

- Source: [KI-Campus public educational video](PUBLIC_SAMPLE_ZH.md), about 97 seconds; media and generated captions were stored locally and are not committed.
- System: local CPU `faster-whisper` base transcription followed by RewindAI's transcript-only keyword retrieval. There was no image understanding or generated answer.
- Questions: three answerable candidates with reference time ranges, three unrelated questions, and three harder supposedly unanswerable questions sharing medical or AI vocabulary. The questions and labels require manual video review.
- Rule: the first returned segment is correct only if it covers at least 50% of the annotated reference interval. Wrong evidence for an answerable question counts as one false positive and one false negative. A returned result for an unanswerable question is a false positive. A correct refusal is a true negative. The other answerable failures are false negatives.

## Observed result

On this one video and nine draft questions: TP = 3, FP = 3, FN = 0, TN = 3; question-level temporal-evidence precision = **0.50**, recall = **1.00**, and F1 = **0.67**. All three false positives were the harder unanswerable candidates. These values do not show whether the answer text is correct and must not be used as portfolio accuracy figures or generalized to other videos.

## Reproduce

Obtain the video as described in [PUBLIC_SAMPLE_ZH.md](PUBLIC_SAMPLE_ZH.md), generate a local subtitle with [LOCAL_TRANSCRIPTION.md](LOCAL_TRANSCRIPTION.md), then run:

```powershell
python scripts/diagnose_transcript.py --video "PATH_TO_VIDEO.webm" --subtitles "PATH_TO_GENERATED.srt" --questions data/evaluation/public_sample_questions.pending.json
```

Different transcription/model versions can change outputs. Next, manually verify video evidence and negative labels, create questions from several independent videos, reserve videos not used during development, and report question-level and retrieval metrics separately. Include false-positive examples and human assessment of answer support.

## Transcription comparison

The same public video was transcribed on CPU using faster-whisper `base` and compared with the video's supplied English subtitles using `scripts/evaluate_transcription.py`. After case normalization and punctuation removal, the reference has 236 tokens and the hypothesis 231. Minimum-edit alignment yielded 228 matches, 3 substitutions, 5 deletions and 0 insertions: **word error rate 3.39%**, aligned-token precision **98.70%**, recall **96.61%** and F1 **97.64%**. Precision is matches divided by hypothesis tokens; recall is matches divided by reference tokens. WER counts substitutions, deletions and insertions divided by reference tokens, and can exceed 100%.

This is a single short, curated public sample. The supplied captions are a convenient reference, not independently verified ground truth; timing, speaker names and semantic correctness were not evaluated. English words and individual CJK characters are used as tokens, so results across languages are not directly comparable. The test is **not** an accuracy claim about question answering or Canvas lectures.

```powershell
python scripts/evaluate_transcription.py --reference "PATH_TO_SUPPLIED.srt" --hypothesis "PATH_TO_GENERATED.srt"
```

