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


## YTCommentQA: independent snippet-level retrieval proxy

We evaluated the keyword transcript pipeline against independently collected [YTCommentQA](https://github.com/lgresearch/YTCommentQA) labels from Yang et al., [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29906). The repository describes naturally occurring instructional-video questions, caption evidence IDs, and labels for answerability from visual, script, combined, or neither; it lists the dataset repository under the MIT license. Video and transcript rights may differ. We obtained the public `dataset.json` locally; no videos, audio or Canvas data were downloaded or uploaded. The source file had Git blob SHA-1 `5298bd108374c10bde5357048bcccccc8ffbc596` and SHA-256 `b6cd61581c2e5e2e15b0ed8ee0bd6e6d45abdd5ba7af5558cf1d5f9bb0dcc368`.

This is an **exploratory proxy**, not a question-answering or video-wide answerability benchmark. Each question is given only the dataset's annotated caption snippet, with each caption sentence indexed as a separate segment. We predict positive whenever keyword retrieval returns any segment, and negative when it returns none. We compare this heuristic against questions labeled script-answerable (including those also answerable visually) versus questions labeled unanswerable in the annotation context. We exclude visual-only and combined-required questions because they cannot be treated as negative for the same task. The authors annotated snippets around reply timestamps; a snippet-level unanswerable label does not establish that an entire video has no answer. A returned keyword match does not establish that the snippet supports an answer. The app normally groups subtitle cues differently, and the original videos and source audio were not inspected.

For a deterministic sample, compute SHA-256 of each video ID and include it when the first eight bytes as a big-endian integer are divisible by five. This keeps questions from the same video together; it is our sampling rule, not an official test split. Of 446 selected records, 145 visual-only/combined-only records were excluded, 17 had invalid evidence-caption time/text values, and **284 questions from 239 videos** remained. No setting was tuned on this sample during this run.

| Snippet-level proxy count | Value |
| --- | ---: |
| True positives / false positives | 92 / 6 |
| False negatives / true negatives | 131 / 55 |
| Precision / recall / F1 | **0.939 / 0.413 / 0.573** |
| Unanswerable snippets returning a candidate | 6 / 61 |

The positive prevalence is high and this deliberately simplified proxy has low recall. The values are not answer accuracy, full-video retrieval accuracy, model-calibrated confidence, or comparable to the paper's answerability classifiers. These are public YouTube how-to material rather than university lectures, and model pretraining overlap is unknown. The exclusion of invalid cues may introduce bias. Even when the heuristic agrees with an answerability label, no answer is generated or checked. Use independent, reviewed course-specific and visual tests before making portfolio performance claims.

To reproduce, obtain the repository's public `dataset.json` separately and keep it outside version control, then run:

```powershell
python scripts/evaluate_ytcommentqa.py --dataset PATH_TO_LOCAL_DATASET_JSON --sample sample
```

`other` and `all` are available for exploration. The script reads the dataset locally and outputs aggregate counts only; model/video downloads and private material are not involved. Attribution: Saelyne Yang, Sunghyun Park, Yunseok Jang and Moontae Lee, *YTCommentQA: Video Question Answerability in Instructional Videos*, AAAI 2024, DOI 10.1609/aaai.v38i17.29906.


### Paired semantic versus keyword proxy on the same snippets

We repeated the same fixed sample and exclusions using local CPU `Xenova/all-MiniLM-L6-v2` quantized ONNX revision `08a308f628bc9d6774b7922f319eb1b65afa1a82`. The semantic path used the pre-existing, uncalibrated cosine cutoff of 0.4; no cutoff was selected on this sample. Both systems saw the same annotated caption snippets, one sentence per indexed segment, and both used the presence of any returned candidate as a positive proxy.

| Method | TP | FP | FN | TN | Precision | Recall | F1 | Unanswerable snippets returning candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword | 92 | 6 | 131 | 55 | 0.939 | 0.413 | 0.573 | 6 / 61 |
| Experimental semantic | 137 | 8 | 86 | 53 | 0.945 | 0.614 | 0.745 | 8 / 61 |

Among 223 script-answerable snippets, both returned candidates for 73, only keyword for 19, only semantic for 64, and neither for 67. Among the 61 snippet-level unanswerable questions, both returned candidates for 3, only keyword for 3, only semantic for 5, and neither for 50. These are paired presence/absence counts, not judgments that the returned text supports the answer. Semantic retrieval returned candidates on more positives and on two more negatives in this sample; there is no inference here about statistical significance or user benefit.

The test is exploratory: the labeled evidence window is supplied in advance, making the task easier and different from finding it in a whole video. False positive means a candidate was returned on a snippet labeled unanswerable; false negative means no candidate was returned on a script-answerable snippet. The source questions are YouTube instructional comments rather than lectures, labels can be imperfect, and prior pretraining overlap is unknown. The excluded 145 visual-only/combined-only cases and 17 invalid-caption cases remain excluded. These metrics do not measure answer correctness, full-video retrieval, visual reasoning or real lecture performance.

```powershell
python scripts/evaluate_ytcommentqa.py --dataset PATH_TO_LOCAL_DATASET_JSON --sample sample --compare-model PATH_TO_LOCAL_MINILM --semantic-threshold 0.4
```

The script uses local files and prints aggregate statistics; it does not download or process video. Reusing this sample to select thresholds would invalidate any later claim that the same sample was untouched.

### Query latency smoke check

On 5 November 2026, the paired local comparison was rerun on the nine provisional ethics-sample questions with the generated captions and quantized MiniLM model described above. In this single run, keyword query time had a median of **0.13 ms** and nearest-rank P95 of **0.29 ms**; semantic query time had a median of **17.51 ms** and P95 of **19.25 ms**. Each method processed nine questions sequentially, with the keyword method run first. These times measure only the in-process query and result serialization after both indexes were built; model loading, indexing, transcription, browser and network latency are excluded. With only nine observations, P95 is the maximum, and these values are not a general performance benchmark. The draft labels remain unverified and this smoke check does not change the earlier quality conclusions.


### Searching the provided transcript for annotated time windows

To move closer to the actual retrieval task, `scripts/evaluate_ytcommentqa_full_transcript.py` searches all caption sentences supplied for each question's video, rather than supplying only the annotated snippet. It evaluates questions labeled script-answerable (including those also visually answerable); other modality labels and snippet-level unanswerable labels are excluded. No source video or audio is used. The reference window is the time span from the earliest to latest caption in the dataset's annotated evidence range. We count a hit when a returned sentence has **any positive-duration overlap** with that window. The top-three score is a question-level overlap rate, not precision or answer F1; the data do not establish every other relevant interval.

On the same deterministic 20% video-ID sample, the strict run excluded any question with an invalid caption anywhere in its supplied transcript: 446 selected records included 218 other-label exclusions and 77 invalid transcripts, leaving **151 questions from 126 videos**. Both methods used identical questions, source captions and exclusions; the semantic ONNX model and uncalibrated 0.4 threshold were the same as above. A simple reference baseline selects the first three available captions in time order.

| Method | Top-one overlap | Top-three overlap | No candidate |
| --- | ---: | ---: | ---: |
| Keyword | 31 / 151 (0.205) | 50 / 151 (0.331) | 55 / 151 |
| Experimental semantic | 53 / 151 (0.351) | 80 / 151 (0.530) | 30 / 151 |
| First three captions, no search | — | 11 / 151 (0.073) | — |

For top-three overlap, both methods overlapped for 35 questions, only keyword for 15, only semantic for 45, and neither for 56. This is an exploratory paired comparison, not a significance test. An overlapping sentence may cover only a tiny part of a broad evidence window and may not answer the question; annotations can omit other valid locations. The prior snippet experiment used this same sample, so it is not an untouched final test. Caption sentence segmentation differs from the application's grouped subtitle segments, the videos are public YouTube instructional material rather than university lectures, and model pretraining overlap is unknown.

Because 77 exclusions can introduce bias, an additional sensitivity run skipped invalid captions **outside** each valid reference range rather than rejecting the entire transcript. It retained **223 questions from 186 videos**, still excluding five with invalid reference captions; 72 retained transcripts were incomplete and 113 invalid cues were omitted. Keyword top-three overlap was **75 / 223 (0.336)**, semantic **117 / 223 (0.525)**, and the first-three baseline **12 / 223 (0.054)**. Missing cues can remove legitimate competing evidence and change ranks, so this does not establish the result on complete transcripts. The two runs have different denominators; neither represents answer accuracy, semantic correctness, visual performance or full-video unanswerable rejection.

```powershell
python scripts/evaluate_ytcommentqa_full_transcript.py --dataset PATH_TO_LOCAL_DATASET_JSON --sample sample --compare-model PATH_TO_LOCAL_MINILM --semantic-threshold 0.4
# Optional sensitivity run: append --allow-incomplete-captions
```

The local dataset and model files must already exist. The script prints aggregate counts only and does not download media. Reproduce with the data and model revisions documented above; independently annotated lecture videos and actual answer evaluation remain necessary.


### Reference-window coverage in the strict transcript run

The earlier overlap rule counts even a very short intersection. We repeated the same strict 151-question run and additionally clipped each returned segment to its annotated reference window, merged overlapping clips to avoid double-counting, and measured the fraction of that window covered by up to three returned segments. No-candidate cases contribute zero coverage.

| Method | At least half of reference window covered | Mean reference coverage | Returned segments overlapping window |
| --- | ---: | ---: | ---: |
| Keyword | 16 / 151 (0.106) | 0.134 | 63 / 239 (0.264) |
| Experimental semantic | 20 / 151 (0.132) | 0.183 | 95 / 314 (0.303) |
| First three captions | 7 / 151 (0.046) | 0.044 | Not calculated |

This exposes how a small overlap can coexist with low coverage of a broad annotated window. It is **not** answer correctness: the reference snippet can contain more context than needed, so a shorter clip may still support an answer, while a long overlapping clip may not. The returned-segment overlap fraction is not factual precision, because the dataset may omit other valid answer locations. A maximum of three individually indexed sentences also limits possible coverage, especially of long windows. The 77 excluded invalid transcripts, previously exposed sample, transcript-only setup, differing app segmentation, uncalibrated semantic threshold and YouTube-versus-lecture limitations described above still apply. The 50% cutoff is a diagnostic choice, not a validated success threshold.

Run the same command in the preceding section; the output now contains `top3_reference_half_covered_count`, `mean_top3_reference_coverage` and `returned_segment_window_overlap_fraction`. Coverage checks use the union of the returned time intervals rather than adding overlaps twice.

### Segmentation sensitivity on the same public transcripts

The default evaluation indexes each supplied sentence independently. To approximate the workspace's grouping, `--segmentation workspace` applies the actual subtitle grouping routine to those same time-ordered sentences. On the same strict 151 questions and model settings, the exploratory results were:

| Method and segmentation | Top-three overlap | At least half of reference window covered | Mean reference coverage | No candidate |
| --- | ---: | ---: | ---: | ---: |
| Keyword, sentence | 50 / 151 (0.331) | 16 / 151 (0.106) | 0.134 | 55 |
| Keyword, workspace grouping | 61 / 151 (0.404) | 35 / 151 (0.232) | 0.235 | 48 |
| Semantic, sentence | 80 / 151 (0.530) | 20 / 151 (0.132) | 0.183 | 30 |
| Semantic, workspace grouping | 74 / 151 (0.490) | 36 / 151 (0.238) | 0.278 | 40 |

The grouping changes the searchable text and the length of each returned interval; it can also span gaps between captions. Longer windows can therefore raise measured coverage without providing additional relevant words. The first-three baseline likewise selects grouped segments in workspace mode, despite the legacy `first_three_caption` output name. Source sentences are not necessarily the cues produced by actual SRT or speech recognition, and no actual video or lecture was processed. These metrics do not establish answer correctness, precision, recall, F1, or a general improvement. The sample was already explored; it must not be treated as an untouched test for choosing a segmentation strategy.

```powershell
python scripts/evaluate_ytcommentqa_full_transcript.py --dataset PATH_TO_LOCAL_DATASET_JSON --sample sample --compare-model PATH_TO_LOCAL_MINILM --segmentation workspace
```
