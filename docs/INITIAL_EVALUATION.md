# Initial local retrieval diagnostic

This is a **small provisional engineering diagnostic**, not a validated benchmark or a measure of answer accuracy. The video and reference subtitles are public; the nine questions in [`public_sample_questions.pending.json`](../data/evaluation/public_sample_questions.pending.json) are draft labels, have not been independently verified against the video, and were designed while developing the system. They are not a held-out test set.

## Setup

- Source: [KI-Campus public educational video](PUBLIC_SAMPLE_ZH.md), about 97 seconds; media and generated captions were stored locally and are not committed.
- System: local CPU `faster-whisper` base transcription followed by RewindAI's transcript-only keyword retrieval. There was no image understanding or generated answer.
- Questions: three answerable candidates with reference time ranges, three unrelated questions, and three harder supposedly unanswerable questions sharing medical or AI vocabulary. The questions and labels require manual video review.
- Rule: the first returned segment is correct only if it covers at least 50% of the annotated reference interval. Wrong evidence for an answerable question counts as one false positive and one false negative. A returned result for an unanswerable question is a false positive. A correct refusal is a true negative. The other answerable failures are false negatives.

## Observed result

On this one video and nine draft questions: TP = 3, FP = 3, FN = 0, TN = 3; question-level temporal-evidence precision = **0.50**, recall = **1.00**, and F1 = **0.67**. All three false positives were the harder unanswerable candidates. These values do not show whether the answer text is correct and must not be used as portfolio accuracy figures or generalized to other videos.

### Experimental semantic retrieval

An additional CPU experiment used [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), quantized ONNX model revision `08a308f628bc9d6774b7922f319eb1b65afa1a82`, with a deliberately uncalibrated cosine cutoff of 0.4. The same nine provisional questions gave TP = 2, FP = 3, FN = 1, TN = 4: precision **0.40**, recall **0.67**, F1 **0.50**. This is below the keyword baseline on this tiny sample, so it has **not** replaced the default retrieval. These are not held-out results; the cutoff is illustrative rather than validated, and this model is English-only. Cosine similarity is not a calibrated probability and cannot establish answerability.

To repeat, install `pip install -e ".[semantic]"`, obtain that model revision's `tokenizer.json` and `onnx/model_quantized.onnx` into a local directory, then use `python scripts/diagnose_semantic.py --video VIDEO --subtitles SRT --questions data/evaluation/public_sample_questions.pending.json --model-dir LOCAL_MODEL_DIRECTORY --threshold 0.4`. The diagnostic itself reads model and video data locally; the model files and source videos are not committed. Model weights have their own Apache-2.0 licence. Do not tune a threshold on the final test set.

### Second-video diagnostic

A second [public KI-Campus video](PUBLIC_SAMPLE_ZH.md) was transcribed locally with the same speech model. Ten new questions, five answerable and five intended as unanswerable, were drafted from its published captions; their times and labels have not been independently verified. Using the same 50% reference-coverage rule and the previously selected semantic cutoff of 0.4:

| Method | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword | 2 | 5 | 3 | 3 | 0.286 | 0.400 | 0.333 |
| Experimental semantic | 3 | 5 | 2 | 2 | 0.375 | 0.600 | 0.462 |

The second video was not part of the first nine-question diagnostic, but the new questions were drafted with knowledge of the system and are not an independent test set. Wrong evidence on an answerable question counts as both FP and FN, so these counts sum to more than ten. The results vary across the two videos and show substantial false positives; neither approach establishes that an answer is supported. Labels are in [`ai_explained_questions.pending.json`](../data/evaluation/ai_explained_questions.pending.json). No further threshold tuning was performed on these questions.

## Reproduce

Obtain the video as described in [PUBLIC_SAMPLE_ZH.md](PUBLIC_SAMPLE_ZH.md), generate a local subtitle with [LOCAL_TRANSCRIPTION.md](LOCAL_TRANSCRIPTION.md), then run:

```powershell
python scripts/diagnose_transcript.py --video "PATH_TO_VIDEO.webm" --subtitles "PATH_TO_GENERATED.srt" --questions data/evaluation/public_sample_questions.pending.json
```

Different transcription/model versions can change outputs. Next, manually verify video evidence and negative labels, create questions from several independent videos, reserve videos not used during development, and report question-level and retrieval metrics separately. Include false-positive examples and human assessment of answer support.

## Transcription comparison

The same public video was transcribed on CPU using faster-whisper `base` and compared with the video's supplied English subtitles using `scripts/evaluate_transcription.py`. After case normalization and punctuation removal, the reference has 236 tokens and the hypothesis 231. Minimum-edit alignment yielded 228 matches, 3 substitutions, 5 deletions and 0 insertions: **word error rate 3.39%**, aligned-token precision **98.70%**, recall **96.61%** and F1 **97.64%**. Precision is matches divided by hypothesis tokens; recall is matches divided by reference tokens. WER counts substitutions, deletions and insertions divided by reference tokens, and can exceed 100%.

This is a single short, curated public sample. The supplied captions are a convenient reference, not independently verified ground truth; timing, speaker names and semantic correctness were not evaluated. English words and individual CJK characters are used as tokens, so results across languages are not directly comparable. The test is **not** an accuracy claim about question answering or Canvas lectures.

As an additional check, the same local model was run on a second [public KI-Campus video](PUBLIC_SAMPLE_ZH.md), about 143 seconds long. Its published English subtitle contained 282 reference tokens; the generated transcript also contained 282. Alignment found 277 matches, 5 substitutions, no insertions or deletions: WER **1.77%**, precision **98.23%**, recall **98.23%** and F1 **98.23%**. This second sample has the same producer and style; its reference is likewise not independently verified. The two examples are too few to establish general performance, especially for real lectures with background noise, accents or specialist vocabulary.

```powershell
python scripts/evaluate_transcription.py --reference "PATH_TO_SUPPLIED.srt" --hypothesis "PATH_TO_GENERATED.srt"
```

## Supplied captions versus local speech transcription

As a paired input-sensitivity diagnostic, the same keyword retriever and provisional questions were run on both the published English captions and the locally generated faster-whisper `base` captions for each video. All other diagnostic settings were unchanged; metrics use the top-1 evidence and 50% reference-interval coverage rule described above.

| Video | Caption input | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AI medical ethics (9 questions) | Published captions | 3 | 3 | 0 | 3 | 0.500 | 1.000 | 0.667 |
| AI medical ethics (9 questions) | Local Whisper base | 3 | 3 | 0 | 3 | 0.500 | 1.000 | 0.667 |
| AI explained (10 questions) | Published captions | 4 | 2 | 1 | 3 | 0.667 | 0.800 | 0.727 |
| AI explained (10 questions) | Local Whisper base | 2 | 5 | 3 | 3 | 0.286 | 0.400 | 0.333 |

For example, on the second video the question about which tasks AI can do retrieved a top segment at 4.294–24.036 seconds with published captions, covering its 16.302–21.306 second draft reference; the generated captions yielded 0–14 seconds, which does not cover it. The question about how programmers provide instructions instead produced no result with published captions and a matching temporal segment with generated captions. These observations measure timestamp overlap only: they do not independently establish whether an answer is correct.

The reference and generated subtitle files also differ in timing and cue boundaries, which determine how RewindAI groups text into segments. The comparison therefore combines differences in text, timing and segmentation; it cannot isolate transcription error as the cause. Questions and reference intervals were drafted with access to the supplied captions, creating potential label bias, and have not been independently checked. Both videos come from the same producer. The large swing on ten draft questions should not be reported as a population-level accuracy or a product benchmark. No threshold or retrieval settings were changed after observing the results.

To repeat, run the following twice per video, first with the downloaded supplied captions and then with locally generated captions. Keep the same video and question file for both runs. The generated-caption procedure is in [LOCAL_TRANSCRIPTION.md](LOCAL_TRANSCRIPTION.md).

```powershell
python scripts/diagnose_transcript.py --video PATH_TO_VIDEO --subtitles PATH_TO_CAPTIONS --questions data/evaluation/ai_explained_questions.pending.json
```

For the ethics video use `data/evaluation/public_sample_questions.pending.json`. Preserve the JSON output locally for per-question returned ranges and coverage; avoid publishing private videos, subtitles, questions or output. A stronger next test requires independently reviewed time intervals and answerability labels on held-out videos, plus a controlled experiment that fixes segmentation while changing transcript text.

### Controlled window-boundary diagnostic

To investigate segmentation sensitivity, we repeated the same provisional keyword evaluation using `scripts/compare_caption_inputs.py`. Both caption inputs were assigned to the same 20-second windows starting at zero; each whole caption cue was assigned to the window containing its midpoint. Empty windows were retained on both sides. This is an experimental comparison, not the application's default grouping.

| Video | Caption input | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AI medical ethics | Published | 3 | 3 | 0 | 3 | 0.500 | 1.000 | 0.667 |
| AI medical ethics | Whisper base | 3 | 3 | 0 | 3 | 0.500 | 1.000 | 0.667 |
| AI explained | Published | 4 | 2 | 1 | 3 | 0.667 | 0.800 | 0.727 |
| AI explained | Whisper base | 4 | 3 | 1 | 3 | 0.571 | 0.800 | 0.667 |

The second video's generated-caption F1 changed from 0.333 under the default cue grouping to 0.667 under these fixed windows, while the published-caption score stayed at 0.727. This suggests segmentation choices deserve further investigation; it is not proof that segmentation alone caused the earlier difference. Caption content, timing errors, midpoint assignment, ranking, and fixed boundaries can still affect results, and a long cue can straddle a boundary despite being assigned wholly to one window. Twenty seconds was a diagnostic choice and was not selected on an independent validation set. All earlier caveats about provisional, caption-derived labels and tiny samples apply.

```powershell
python scripts/compare_caption_inputs.py --video PATH_TO_VIDEO --reference-captions PATH_TO_SUPPLIED.srt --generated-captions PATH_TO_GENERATED.srt --questions data/evaluation/ai_explained_questions.pending.json --window-seconds 20
```

For the first video use `data/evaluation/public_sample_questions.pending.json`. The JSON output includes per-question ranges and the counts behind each metric. The tool processes the input files locally and does not upload them.

### Experimental semantic workspace smoke test

The optional semantic mode was also run locally with the same quantized MiniLM model and generated captions for the public ethics sample. The HTTP service reported `semantic_experimental`. At the illustrative threshold of 0.4, `medical professionals` abstained; `How can AI help doctors select therapy?` returned a first segment at 36–56 seconds. These two observations check serving and expose uncertain retrieval behavior; they do not establish answer correctness or provide new accuracy metrics. The mode does not replace the default keyword baseline.

## Independent TutorialVQA transcript benchmark

We evaluated the existing keyword retrieval baseline on [TutorialVQA](https://github.com/acolas1/TutorialVQAData), an independently created instructional-video question answering dataset by Colas et al. ([LREC 2020 paper](https://aclanthology.org/2020.lrec-1.670/)). The dataset authors license their data under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The test set was not authored by RewindAI; however, it covers Photoshop screencast tutorials rather than university lectures. It contains some low-quality questions and all items are treated as answerable; results therefore do not assess unanswerable-question rejection.

On the original `test.json` (Git blob `de3b59001325df56c33c6180d62353054a5ebfbe`) and `videos.json` (Git blob `583ce560cf51aba22d78f4385e2bd7403049945d`), all 1,239 questions and 76 referenced videos passed range validation. We indexed each supplied transcript sentence as a separate segment, selected only the question's known source video, and returned at most three sentences using the existing default keyword retriever and reranker. We interpreted the dataset's start and end sentence indices as inclusive. The segment start/end fields in this experiment are **sentence indices, not video seconds**. No source video was downloaded or processed.

| Measure | Result |
| --- | ---: |
| True-positive retrieved sentences | 1,028 |
| False-positive retrieved sentences | 1,207 |
| False-negative reference sentences | 7,312 |
| Micro sentence precision | 0.460 |
| Micro sentence recall | 0.123 |
| Micro sentence F1 | 0.194 |
| Top-1 sentence falls within reference span | 554 / 1,239 = 0.447 |
| Questions with no returned evidence | 273 / 1,239 = 0.220 |

Counts are aggregated over question–sentence pairs; precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. Reference spans often include multiple sentences while the system returns at most three, limiting achievable recall; this experiment is not directly comparable to full answer-span prediction results in the paper. A sentence within a reference span is treated as relevant without independent semantic review, and source transcript quality and the inclusive-index interpretation may affect scores. These numbers are **not** video-timestamp precision/recall, answer accuracy, speech-recognition performance, or a measure of the full multimodal system. Test data were not used to choose or tune the retrieval settings for this run.

To reproduce, obtain `videos.json` and `test.json` directly from the linked TutorialVQA repository, observe its noncommercial data licence and citation requirements, then run locally:

```powershell
python scripts/evaluate_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_TEST_JSON
```

The external data are not bundled in RewindAI. This script does not access the internet.

### Adjacent sentence context experiment

We explored whether including neighboring transcript sentences around each returned hit improves reference-span coverage. `scripts/evaluate_tutorialvqa.py --context-sentences N` expands each top-three retrieved sentence by N sentences on each side, clips to the transcript and deduplicates overlaps. It changes only the evaluation output set: the initial ranking, top-1 hit and abstention behavior are unchanged. This experiment does not modify the application.

On the dataset's 1,238-question development split (`dev.json`, Git blob `f9901f4bc5bf3c131ba98c64d89776a75431909e`):

| Context on each side | Micro precision | Micro recall | Micro F1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.477 | 0.135 | 0.210 |
| 1 | 0.394 | 0.294 | 0.337 |
| 2 | 0.350 | 0.402 | 0.374 |
| 3 | 0.317 | 0.479 | 0.382 |

Among these four tested settings, three adjacent sentences gave the highest development F1. Applied to the 1,239-question test split, it produced TP 3,778, FP 8,373 and FN 4,562: precision **0.311**, recall **0.453** and F1 **0.369**, compared with **0.460 / 0.123 / 0.194** without context. More text covers more reference sentences but also introduces more unrelated sentences; it does not improve the initial hit. The test still has 273 abstentions and 554 top-1 overlaps.

This is an exploratory comparison, not a generalization result: we inspected the baseline test result before this experiment, all 76 test videos also appear in the development split, and 12 distinct `(video_id, question)` pairs appear in both splits (exact string comparison). Context size was selected from only the four values 0–3 on development data. The data's answer spans are treated as inclusive; the reference transcripts, not model-transcribed lecture audio, are evaluated. Production uses variable-length timed subtitle chunks rather than individual sentence indices, so these numbers cannot be assumed to carry over. Evaluate on genuinely unseen videos and review relevance before deciding whether to expose extra context in the product.

```powershell
python scripts/evaluate_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_DEV_OR_TEST_JSON --context-sentences 3
```

### Video-disjoint exploratory split

The original TutorialVQA development and test files share videos. To avoid overlap between the videos used for choosing the context setting and those used for the next comparison, the evaluator now supports `--video-partition`. It hashes the UTF-8 video ID with SHA-256, interprets the first eight bytes as a big-endian integer, and assigns IDs divisible by five to `heldout`; all others are `development`. This partition is deterministic and does not use question labels.

From the original development file, the development partition contains 936 questions from 58 videos. Among context sizes 0, 1, 2 and 3, micro F1 was respectively 0.216, 0.344, 0.383 and 0.388; we selected 3. The heldout partition of the original test file contains 294 questions from 18 videos, none shared with the development partition. Its results:

| Context on each side | TP | FP | FN | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 211 | 324 | 2,007 | 0.394 | 0.095 | 0.153 |
| 3 | 828 | 2,140 | 1,390 | 0.279 | 0.373 | 0.319 |

Both settings have 110 top-1 hits and 66 abstentions: surrounding context adds coverage but does not change retrieval ranking. The numbers reflect supplied transcript sentences, not video timestamps or answer correctness. Crucially, these are **not a pristine held-out benchmark**: the earlier experiment inspected aggregate results from all videos, including these 18, and the video partition was created afterward. No per-video data were excluded from the earlier aggregate. A future untouched video set and independently reviewed timestamps are still required for confirmatory claims.

```powershell
python scripts/evaluate_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_DEV_JSON --video-partition development --context-sentences 3
python scripts/evaluate_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_TEST_JSON --video-partition heldout --context-sentences 3
```

### Keyword versus local semantic retrieval

We ran the existing experimental CPU semantic mode on the same 294 questions from 18 video-disjoint test-side videos used above, with **no adjacent-sentence expansion**. The model was `Xenova/all-MiniLM-L6-v2`, quantized ONNX revision `08a308f628bc9d6774b7922f319eb1b65afa1a82`. The cosine cutoff remained the previously used, uncalibrated 0.4; it was not tuned on this subset. Each method searched the same supplied transcript for the known source video and returned up to three sentences.

| Method | TP | FP | FN | Precision | Recall | F1 | Top-1 in reference | Abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword | 211 | 324 | 2,007 | 0.394 | 0.095 | 0.153 | 110 / 294 | 66 |
| Experimental semantic | 287 | 453 | 1,931 | 0.388 | 0.129 | 0.194 | 136 / 294 | 27 |

On this sample, semantic retrieval returned more reference sentences and more non-reference sentences. These are micro sentence-overlap metrics against the dataset's inclusive reference ranges, not semantic correctness or video-time localization. All questions are treated as answerable, so fewer abstentions do not establish that semantic retrieval handles unanswerable questions safely. This subset is video-disjoint from the partition used in the adjacent-context experiment, but was included in earlier aggregate analysis; it is not a pristine held-out benchmark. The source consists of Photoshop instructional transcripts, not real lecture recordings or ASR output. The difference is exploratory, with no statistical significance or general performance claim.

```powershell
python scripts/evaluate_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_TEST_JSON --video-partition heldout --semantic-model PATH_TO_LOCAL_MODEL_DIRECTORY --semantic-threshold 0.4
```

Install the optional semantic dependencies and supply the model files as described in the README. This evaluation reads input and model files locally.

### Video-level uncertainty diagnostic

To show how much the keyword result can vary with the videos included, we resampled the 18 test-side videos with replacement 2,000 times, keeping all questions belonging to each sampled video together. With seed 42, the 2.5th–97.5th percentile ranges were precision **0.312–0.478**, recall **0.075–0.131**, and F1 **0.122–0.204**; the original point estimates were 0.394, 0.095, and 0.153. Each replicate aggregates sentence TP/FP/FN before computing metrics.

These are exploratory bootstrap ranges for this small, non-random sample of Photoshop tutorials; they are not guaranteed population confidence intervals, do not correct label errors or previous test-set exposure, and do not assess differences between models. All earlier restrictions on sentence-index measurements apply. The fixed seed and number of replicates make the calculation reproducible:

```powershell
python scripts/evaluate_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_TEST_JSON --video-partition heldout --bootstrap-samples 2000 --seed 42
```


### Paired comparison of the two retrievers

We repeated the keyword and semantic comparison on the same 294 questions from the same 18 videos, preserving the 0.4 semantic threshold and no context expansion. The point estimates and sentence counts exactly matched the earlier run. To examine variability in the **difference**, `scripts/compare_tutorialvqa.py` resampled the 18 videos as paired units 2,000 times (seed 42); all questions for a sampled video and both systems' results were kept together. Each resample aggregates sentence TP, FP and FN before computing each metric. The difference is semantic minus keyword:

| Sentence overlap metric | Difference | Exploratory 2.5th–97.5th percentile range |
| --- | ---: | ---: |
| Precision | -0.007 | -0.063 to 0.049 |
| Recall | +0.034 | +0.015 to +0.059 |
| F1 | +0.041 | +0.014 to +0.071 |

These are paired bootstrap ranges for this small, previously examined, non-random set of Photoshop tutorials. They do not correct biased labels, establish statistical significance or prove that semantic retrieval is better on lectures. The sentence-level reference spans and known-video search are not end-to-end answer accuracy; the absence of unanswerable questions remains a key limitation. Semantic retrieval also returned more non-reference sentences (453 versus 324), so the modest F1 difference should not be interpreted without that cost.

```powershell
python scripts/compare_tutorialvqa.py --videos PATH_TO_VIDEOS_JSON --questions PATH_TO_TEST_JSON --semantic-model PATH_TO_LOCAL_MODEL_DIRECTORY --video-partition heldout --semantic-threshold 0.4 --samples 2000 --seed 42
```

The tool processes inputs locally and prints aggregate counts and paired differences, without writing questions or videos. Model and data dependencies retain their own licences.


### Temporal hits and abstentions reported separately

The local diagnostic now also separates temporal evidence coverage on answerable questions from abstention on unanswerable questions, with explicit counts and denominators. On the same nine **unreviewed, caption-derived draft questions** for the public ethics video and the locally generated subtitles, three of three answerable drafts had a top-one segment covering at least half of the draft reference interval; all three also had such a segment among the returned results. There were zero answerable abstentions. Of the six intended unanswerable drafts, three were rejected and three returned candidates. This is the same provisional sample previously reported above, not an independent new test or answer-accuracy measurement.

Optional `question_type` labels allow per-type reporting; missing labels appear as `unlabeled`. The returned-set hit rate is based on the results the pipeline actually returned, not a standard Recall@K over all relevant intervals. A zero denominator yields `null`. Question text and results remain local for private media.


### Paired question-level comparison on both provisional videos

The new local `scripts/compare_time_retrieval.py` runs the actual keyword and experimental semantic pipelines on the same transcript segments and questions, and reports per-question temporal results alongside paired counts. A top-one hit requires at least 50% coverage of the single draft reference interval; it does not establish that the returned text answers the question. The semantic threshold is the earlier uncalibrated 0.4, and the quantized MiniLM model revision is `08a308f628bc9d6774b7922f319eb1b65afa1a82`. Both inputs use locally generated Whisper base captions.

| Public video | Answerable drafts | Both hit | Keyword only | Semantic only | Neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI medical ethics | 3 | 2 | 1 | 0 | 0 |
| AI explained | 5 | 2 | 0 | 1 | 2 |

| Public video | Intended unanswerable drafts | Both abstain | Keyword only abstains | Semantic only abstains | Neither abstains |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI medical ethics | 6 | 3 | 0 | 1 | 2 |
| AI explained | 5 | 2 | 1 | 0 | 2 |

These are the same previously examined caption-derived drafts and the same two videos, not new or independent tests. They were not independently reviewed against video and all questions come from one publisher. Some returned evidence may cover the draft interval yet fail to support an answer. The counts are exploratory and cannot establish superiority or general lecture performance. The detailed JSON contains questions and candidate timestamps; keep any private-course results local.

```powershell
python scripts/compare_time_retrieval.py --video PATH_TO_VIDEO --subtitles PATH_TO_LOCAL_SRT --questions PATH_TO_QUESTIONS_JSON --model-dir PATH_TO_LOCAL_MINILM --semantic-threshold 0.4 --min-coverage 0.5
```

The script reads inputs locally. Select thresholds on a separate validation set before using an untouched test set, and record source and model versions separately.


To save a local report rather than printing detailed questions, append:

```powershell
--output artifacts/evaluation/comparison.json --aggregate-only --record-hashes
```

The output must be a new JSON file under `artifacts/` or `local_data/`; existing files are not overwritten. `--aggregate-only` omits per-question results, while `--record-hashes` checks video, subtitles, questions, tokenizer and model file SHA-256 before and after the run and records them with the thresholds and UTC time. Hashing long videos can be slow. These hashes and aggregate results can still be sensitive; keep them local. A report does not capture every dependency, hardware setting or code revision, so record those separately before claiming reproducibility.
