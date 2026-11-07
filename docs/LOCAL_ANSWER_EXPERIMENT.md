# Experimental local answer evaluation

The optional answer generator is a research prototype. Retrieval can return irrelevant excerpts and a model can invent or distort details even when it says information is missing. Neither a fluent answer nor a timestamp proves correctness.

I ran Qwen2.5-3B-Instruct Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe` on CPU with `llama-cpp-python` 0.3.36, keyword retrieval, temperature 0, and up to 160 output tokens. On 12 short hand-written synthetic cases, it answered six supported questions and declined six unsupported questions; the 0.5B model invented answers on five of the six unsupported cases. The examples were designed for development and are not a benchmark.

I then ran the actual transcript pipeline on nine existing draft questions for the public [KI-Campus ethics video](PUBLIC_SAMPLE_ZH.md), using locally generated Whisper base subtitles. Three draft questions had reference answers and six were intended as unanswerable. Labels came from captions and have not been independently reviewed against the video. In one failure, asked which diabetes treatment the AI recommended, the model said it recommended a diabetes treatment even though the retrieved text did not mention diabetes, then said the specific treatment was not provided. The generated caveat did not undo the unsupported claim. This example illustrates a limitation, not a prevalence estimate. No numerical answer accuracy or F1 is claimed.

A follow-up on the same nine draft questions used a stricter prompt that explicitly treats question assumptions as unverified and requests a fixed insufficient-evidence response when details are missing. In this run, the diabetes-treatment question received that insufficient-evidence response instead of the earlier unsupported claim; the other two related unanswerable questions that reached the model also received it. The three unrelated questions were rejected by retrieval before generation. The three draft answerable questions still received answers. This is a previously inspected nine-question development sample, not an independent test or evidence of a general reliability improvement. Prompt wording does not prevent hallucinations or prompt injection. The pipeline's `abstained` flag records retrieval abstention only; it does not classify whether generated text declines to answer.

To create a local draft for human review, run:

```bash
python scripts/evaluate_local_answers.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --questions PATH_TO_QUESTIONS_JSON --model PATH_TO_GGUF --output artifacts/local_answer_review/run.json
```

The question JSON uses the same `questions` structure as the prepared temporal question files. The script rejects test-split files, fingerprints inputs before and after execution, and writes only to a new JSON file under `artifacts/` or `local_data/`. A source-bound question file is checked against the supplied video and subtitles; unbound files are marked as such. Results include question text, draft answers, retrieved excerpts, timings, input fingerprints and the recorded generation configuration. Keep private course outputs local. Human answer and evidence verdicts start empty; retrieval abstention is not a judgment that a generated answer is correct. Review the source recording, references and each claim before calculating or presenting metrics.

The `start_answer_review.py` workflow can instead build a review directly from video-bound development or validation CSV labels. It defaults to keyword retrieval; pass `--semantic-model PATH_TO_QWEN3_GGUF` to review the experimental semantic route using its default uncalibrated threshold of 0.4. The resulting draft records the retrieval setting and local model fingerprints. Each draft corresponds to its specific configuration; it does not assess visual retrieval. All labels, fingerprints and course-derived drafts stay local.

### Frozen answer test after development

After fixing the model, retrieval method and thresholds using separate development and validation videos, and actually checking the test questions, reference answers and scopes against the original video, a reviewer may explicitly create a private answer-test manifest:

```powershell
python scripts/freeze_answer_test.py --video PATH_TO_TEST_VIDEO --subtitles PATH_TO_SRT --labels PATH_TO_TEST_CSV --video-id TEST_VIDEO_ID --model PATH_TO_GGUF --semantic-model PATH_TO_QWEN3_GGUF --output artifacts/local_answer_review/frozen-answer-test.json --confirm-labels-reviewed
```

Omit `--semantic-model` for the keyword baseline. Labels must be a local `test` CSV under `artifacts/` or `local_data/` with the labeler's matching source sidecar. The tool checks reference bounds and whole-video-check declarations for negative questions, and stores input, selected code, model and runtime fingerprints plus the chosen retrieval setting. The checked flag is a human declaration—not proof of correct labels, independence or video-disjoint data. The current subtitle hash does not prove those subtitles were used during annotation. Keep the manifest private.

Then run `scripts/start_answer_review.py` with the same video, subtitles, labels, model, semantic options and video ID, plus `--split test --frozen-test-manifest PATH_TO_FROZEN_MANIFEST`. It checks the manifest and configuration before and after generating unreviewed local answer drafts, and passes the manifest to the review page. When resuming with `review_answers.py` or scoring with `score_answer_review.py`, provide the same `--frozen-test-manifest` again; they verify the manifest hash, recorded inputs, retrieval/prompt settings, selected current code and runtime versions. A human must still review every reference, answer and supporting excerpt against the video, and explicitly declare review before the page or scoring script computes question-level precision, recall and F1. The report identifies the test as declared, not independently verified. These checks cannot establish that a different person reviewed the answers, detect near-duplicate videos, cover every dependency, or prevent repeated testing or a user-edited report. Do not tune on the test results; keep private questions, answers, fingerprints and metrics on this computer.
## Optional local development suggestions

`generate_development_questions.py` can draft questions from sampled subtitles with an existing local model, writing a new JSON and readable Markdown file only under ignored local directories. It never changes question labels or runs on a test split. These are model-generated, can hallucinate, inherit transcript errors and sampling bias, and are not independently authored questions. Watch the video, verify every question, answer and time, and select **Model suggestion** as the question origin when entering it in the development labeler. The source marker is carried into answer-review drafts. Do not treat these suggestions as an independent evaluation set or upload private course content. To browse them directly inside the development labeler, add `--candidates PATH_TO_SUGGESTIONS_JSON` when starting `scripts/temporal_label_server.py`; the video and subtitles must match the file fingerprints. Filling a suggestion never saves a label automatically.

```bash
python scripts/generate_development_questions.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --model PATH_TO_GGUF --output artifacts/private_lecture_review/suggestions.json --count 8
```

## Start from local question labels

For labels created in the local temporal-label page, the helper below verifies that the CSV is bound to the same video and that its split is development or validation. It creates answer drafts using CPU inference, then starts the review page. A final test remains reserved unless a frozen-answer-test manifest is explicitly supplied as described above. All inputs and outputs stay local.

```bash
python scripts/start_answer_review.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --labels artifacts/private_lecture_review/labels.csv --video-id VIDEO_ID --split development --model PATH_TO_GGUF
```

Open `http://127.0.0.1:5097/` after processing. Draft questions must exist first. The model may make unsupported claims; a draft and its resulting scores are not independently verified ground truth. Source fingerprints, questions, answers and reports can be private.

## Review without editing JSON

Start the local review page with the same source video used to produce the review draft:

```bash
python scripts/review_answers.py --video PATH_TO_VIDEO --review artifacts/local_answer_review/run.json
```

Open `http://127.0.0.1:5097/` on the same computer. Play the reference and candidate timestamps, check the question and reference, and enter an answer and evidence verdict for every row. If a reference is wrong, correct the original question labels and regenerate the draft instead of assigning a misleading score. For the evidence-grounded score, a factually correct answer without support in the model input is marked incorrect or unsupported. A mere topical match does not establish that all substantive claims are supported; abstention is reserved for a refusal without unsupported claims. Use **Save progress** to write a new local `progress-*.json` copy without calculating metrics; use **Restore saved progress** to recover the latest matching copy. To resume after restarting, run `python scripts/resume_answer_review.py --video PATH_TO_SAME_LOCAL_VIDEO`; it opens the most recently updated matching saved progress or development/validation draft. Frozen test reviews still require the explicit `review_answers.py --review ... --frozen-test-manifest ...` command. The file contains private questions, answers and judgments. New drafts also record the truncated question and each excerpt actually supplied to the local generator, alongside the full retrieved subtitle. Older drafts show these inputs as not recorded; regenerate a draft if you need this distinction, and retain your previous review separately. After reviewing all rows, declare the review and save. A new `reviewed-*.json` file is created beside the input; the original is untouched. The page displays question-level metrics from the declared judgments, plus counts of model-generated questions and questions whose origin was not recorded. Neither count establishes independent authorship. These are private draft results, and neither the judgments nor the reference labels are independently verified. Changes in the source video or report block saving.

**Suggest exact refusals** can prefill unanswered judgments only for the fixed insufficient-evidence response. It does not overwrite existing judgments, recognize other refusal wording or verify the reference; the reviewer must check each suggestion before declaring review.

## Human review and question-level metrics

After creating a review draft, a person must check every reference and generated answer against the recording, including whether the cited evidence supports every claim. Alternatively, in the local JSON, set `human_answer_verdict` to `correct`, `incorrect` or `abstain`, and `human_evidence_verdict` to `supported`, `unsupported` or `not_applicable`. Use `correct` only for an answerable question when both the answer and evidence are supported. If the answer adds an unsupported claim, mark it incorrect even if it contains a caveat. Use `abstain` only if the answer makes no unsupported answer claim, and pair it with `not_applicable`. The `retrieval_abstained` field only describes retrieval and must not automatically determine the judgment.

Once every row and reference has actually been reviewed, save the edited review locally and run:

```bash
python scripts/score_answer_review.py --review artifacts/local_answer_review/run.json --output artifacts/local_answer_review/summary.json --confirm-human-review
```

The flag is a declaration by the reviewer, not proof of independent review. Each correct supported answer is a true positive; an incorrect answer is a false positive and, for an answerable question, also a false negative. An abstention on an answerable question is a false negative; one on an unanswerable question is a true negative. Precision, recall and F1 are micro-averaged across questions; undefined denominators produce `null`. The report also shows the number of answerable and unanswerable questions, answerable abstentions and the unanswerable abstention rate. If no unanswerable questions were included, that rate is undefined and rejection of unsupported questions has not been assessed. The new summary is saved locally without questions or answers, but its hash and metrics can still be private. The script rejects incomplete or inconsistent judgments and test reports lacking the frozen-test declaration. It does not verify the human judgments or turn draft references into a gold standard.

## Lexical answer overlap is a separate diagnostic

The legacy `scripts/evaluate_answers.py` compares generated answer strings against nonempty reference strings. English-like words are treated as word tokens, while CJK ideographs are treated individually; it reports normalized exact match and mean lexical token F1. It cannot assess whether a paraphrase is correct, whether evidence supports the answer, or whether an answer should have been refused. Empty references are rejected; evaluate unanswerable questions separately. It also rejects duplicate question IDs. Its tokenization changed from the earlier character-level implementation, so scores from the two versions must not be compared directly. Private inputs and resulting metrics must stay local.
## Public annotated snippet experiment (development only)

I selected 25 script-answerable and 25 unanswerable examples from the previously examined YTCommentQA video-hash partition, ranking row IDs by SHA-256 within each class. The 50 examples represent 48 videos; visual-only and combined-required labels were excluded. The dataset's human labels concern annotated snippets, not whole-video answerability. The input to the workspace retrieval was the caption text within each question's annotated evidence range, which supplies the relevant region in advance and makes the task easier than whole-video retrieval. No actual source videos or audio were reviewed. This class-balanced subset and previously explored partition are not an independent final test.

Using keyword retrieval and local Qwen2.5 3B Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe` on CPU, prompt `explicit-evidence-v2`, temperature 0 and 160 output tokens, I compared two **answerability proxies**:

| Positive prediction rule | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| At least one retrieved excerpt | 9 | 4 | 16 | 21 | 0.692 | 0.360 | 0.474 |
| Retrieved excerpt and answer differs from exact refusal string | 1 | 0 | 24 | 25 | 1.000 | 0.040 | 0.077 |

The second rule treats retrieval abstention or the exact normalized phrase `Insufficient evidence in the provided excerpts` as negative; every other generated string counts as positive. It does not recognize other refusals and does not judge the answer. Eight positive examples retrieved evidence but produced that exact refusal. Thus, in this small set the stricter behavior reduced positive predictions substantially; the high precision with only one positive prediction must not be presented alone or generalized to lectures. An attempted answer can still be wrong. Labels, snippet selection, English captions, uncalibrated keyword retrieval, truncated excerpts, model pretraining overlap and prior sample exploration limit the comparison. The numbers are not answer correctness, evidence precision, or course F1.

The public dataset is [YTCommentQA](https://github.com/lgresearch/YTCommentQA); the local dataset SHA-256 was `b6cd61581c2e5e2e15b0ed8ee0bd6e6d45abdd5ba7af5558cf1d5f9bb0dcc368`. Reproduce with the locally downloaded dataset and model:

```bash
python scripts/evaluate_ytcommentqa_local_answers.py --dataset PATH_TO_DATASET_JSON --model PATH_TO_GGUF --per-class 25 --output artifacts/local_answer_review/ytcommentqa.json
```

The script performs no download and saves a local aggregate. For private data, fingerprints and metrics are private too. Additional independent, human-reviewed full-video tests are required.
## Same-sample prompt investigation

After inspecting failures on the 50-example subset, I tried a development-only alternative instruction that permits a simple inference grounded in the excerpts while requesting the same exact refusal phrase when the answer is unsupported. Under the same automatic exact-string proxy, the strict default gave TP/FP/FN/TN **1/0/24/25** (precision 1.000, recall 0.040, F1 0.077); the alternative gave **3/1/22/24** (precision 0.750, recall 0.120, F1 0.207). The retrieval stage was identical: 9/4/16/21. Both prompts requested the same refusal string, but a different refusal wording is still treated as a positive, and no generated answers were graded for correctness.

The alternative was designed after looking at this development sample. Its apparent change is not an independent improvement estimate, and neither sample is appropriate for choosing a portfolio claim. I have not changed the workspace default. Reproduce the alternative with the command above plus `--prompt balanced` and a new output path. Use held-out, independently reviewed questions and assess unsupported claims before deciding whether to adopt it.
## Video-disjoint public snippet check

With both prompt texts fixed after the development investigation, I selected 25 script-answerable and 25 unanswerable rows using the same row-ID hash rule from the `other` video-hash partition. These 50 rows came from 50 videos and are video-disjoint from the 20% development partition; available records did not identify a previous run of this prompt experiment on this partition. Dataset examples may still overlap model pretraining, and this is not an independently reviewed full-video or course test. Labels are for annotated snippets, and the supplied evidence range makes retrieval easier than real whole-video search.

| Rule on the video-disjoint sample | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| At least one retrieved excerpt | 8 | 3 | 17 | 22 | 0.727 | 0.320 | 0.444 |
| Strict prompt, exact-refusal proxy | 2 | 0 | 23 | 25 | 1.000 | 0.080 | 0.148 |
| Balanced prompt, exact-refusal proxy | 4 | 1 | 21 | 24 | 0.800 | 0.160 | 0.267 |

This repeats the direction of the development-sample trade-off, but positive counts are only two and four, respectively, and the exact-string heuristic can misclassify refusals phrased differently. These are class-balanced answerability proxies, not generated-answer correctness. Both prompts used the same dataset/model fingerprints and CPU settings described above; no course material was involved. The balanced prompt remains experimental, and the workspace default is unchanged. Repeat with `--partition other` and separate output paths for `--prompt strict` and `--prompt balanced`. Avoid further tuning on this partition; evaluate answer quality and evidence on independently reviewed recordings.
## Semantic retrieval on the existing development sample

I repeated the same 50 balanced development snippets with the local quantized MiniLM encoder revision `08a308f628bc9d6774b7922f319eb1b65afa1a82`, English-only, CPU, and the previously used uncalibrated 0.4 threshold. The Qwen model and prompt configurations were unchanged. The same question and caption snippet were used as in the keyword runs.

| Development-sample rule | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Semantic retrieval: any candidate | 15 | 4 | 10 | 21 | 0.789 | 0.600 | 0.682 |
| Semantic + strict prompt: exact-refusal proxy | 2 | 0 | 23 | 25 | 1.000 | 0.080 | 0.148 |
| Semantic + balanced prompt: exact-refusal proxy | 5 | 1 | 20 | 24 | 0.833 | 0.200 | 0.323 |

The keyword counterparts on this same development set were 9/4/16/21 for retrieval, 1/0/24/25 for strict and 3/1/22/24 for balanced. An exact-refusal match is a narrow heuristic and other refusal wording counts as a positive; the evaluation does not check generated-answer correctness. The selected evidence window is supplied in advance and all earlier caveats about dataset labels, prompt tuning, balanced sampling, pretraining and lack of source-video review apply. These figures are not course performance and do not establish a general improvement. The workspace defaults remain keyword retrieval and the strict prompt.

To reproduce this exploratory run, add `--semantic-model PATH_TO_LOCAL_MINILM` and use a new local output path. No threshold was selected using the video-disjoint partition; further tuning requires new independent data.

## Semantic retrieval on the other video partition

On 5 November 2026, using the same fixed semantic threshold of 0.4 and strict workspace prompt, I repeated the snippet-level proxy on the 25 answerable and 25 unanswerable examples from the `other` video-hash partition described above. These 50 examples came from 50 videos distinct from the development partition, but the partition had already been used for the keyword/prompt comparison and is not a pristine final test.

| Rule | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Semantic retrieval: any candidate | 17 | 5 | 8 | 20 | 0.773 | 0.680 | 0.723 |
| Semantic + strict prompt: exact-refusal proxy | 2 | 0 | 23 | 25 | 1.000 | 0.080 | 0.148 |
| Semantic + experimental balanced prompt: exact-refusal proxy | 8 | 1 | 17 | 24 | 0.889 | 0.320 | 0.471 |

For context, the keyword retrieval proxy on these same examples was 8/3/17/22 (F1 0.444). The selected evidence snippets are provided in advance, class prevalence is artificially balanced, and the exact-refusal rule does not judge whether generated answers are correct or fully supported. In particular, strict precision of 1.000 comes from only two positive predictions and cannot establish factual accuracy. The balanced prompt produced more positive predictions, but one was on an unanswerable snippet and others may contain unsupported claims; its proxy score is not evidence of safer or more accurate answers. It remains experimental, not the workspace default. No video or audio was reviewed and these are not lecture results. Reproduce with `--partition other --semantic-model PATH_TO_LOCAL_MINILM` and separate new output files for `--prompt strict` and `--prompt balanced`.

As a qualitative check, the assistant inspected the first four positive and first four negative examples in this deterministic sample, comparing each output with the excerpts actually provided to the model. In one positive case, the strict prompt refused even though the supplied sentence appeared to contain the requested event; in another, the strict response appeared supported while the balanced prompt refused. A third question requested an explanation, but the returned text did not appear to explain why; the dataset's positive label alone could not establish that the displayed answer context was sufficient. This is an AI inspection of eight non-random examples, not independent human review or a correctness score; the rest of each video and any omitted captions were not examined. The observations help explain why refusal-proxy counts and answer quality can differ.


### Qwen3 retrieval with local answer generation

On 6 November 2026, I ran one fixed pipeline using Qwen3-Embedding-0.6B-Q8_0 retrieval with the uncalibrated 0.4 threshold and the workspace's strict Qwen2.5-3B-Instruct Q4_K_M answer prompt. I selected 25 script-answerable and 25 unanswerable annotated snippets from the previously explored YTCommentQA `sample` partition by SHA-256-ranked question ID within each class. These 50 questions covered 48 videos. The annotated caption snippet was provided in advance, and the sample is balanced by construction. No source video, audio or private course material was used.

A response was treated as positive by the exact-refusal proxy only if retrieval returned evidence and the generated response was not exactly the requested insufficient-evidence phrase. On the 25 labeled answerable snippets, the proxy was positive for **4**, while 20 generated the fixed refusal and one had no retrieved evidence. On the 25 labeled unanswerable snippets, none was proxy-positive: 18 generated the fixed refusal and seven had no retrieved evidence. The resulting proxy counts were TP 4, FP 0, FN 21, TN 25, yielding precision **1.000**, recall **0.160** and F1 **0.276**.

This heuristic does not determine whether the four non-refusal responses were correct, whether refusals in other wording would be detected, or whether source evidence was sufficient. Precision 1.000 is based on only four positive predictions and is not factual accuracy. The small, balanced, previously explored sample is not representative prevalence or an independent test, and the result is not course-video performance. It indicates a need to examine overly cautious answers with human-reviewed references before changing the prompt or threshold.

Dataset SHA-256: `b6cd61581c2e5e2e15b0ed8ee0bd6e6d45abdd5ba7af5558cf1d5f9bb0dcc368`; embedding model SHA-256: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`; answer model SHA-256: `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`. Inputs were fingerprinted before and after the run; inference used llama-cpp-python 0.3.36 on CPU. The later optional progress option does not affect scoring.

```powershell
python scripts/evaluate_ytcommentqa_local_answers.py --dataset PATH_TO_PUBLIC_DATASET_JSON --model PATH_TO_QWEN25_3B_GGUF --semantic-model PATH_TO_QWEN3_EMBEDDING_GGUF --per-class 25 --partition sample --output artifacts/evaluation/public-answer-run.json --progress
```

The script writes a new local aggregate report, never overwrites an existing one, and does not download media. Keep any course-derived inputs, fingerprints and results private.


#### Small qualitative audit of the Qwen3 answer proxy

A subsequent desk inspection of the first five script-answerable development snippets found different possible reasons behind four fixed refusals. Two supplied captions contain relevant clues: one question has a premise contradicted by the transcript, while another appears to require a commonsense category inference not explicitly stated. The other two ask for a demonstration or later steps that may not be fully specified in the supplied snippet. The remaining, very short question received a non-refusal based on a retrieved caption. This inspection used only the supplied captions, not the source videos, and neither the labels nor my judgments were independently verified. It does not establish an error rate or show that every refusal was wrong. Future prompt changes should be checked against video-reviewed answerable and unanswerable examples, including false-premise questions.


### Qwen3 with the experimental balanced prompt: small negative audit

On 7 November 2026, I used the same locally cached Qwen3-Embedding-0.6B Q8 retriever, uncalibrated 0.4 threshold and Qwen2.5-3B Q4 answer model on 20 previously explored public YTCommentQA caption snippets (10 labeled answerable and 10 labeled unanswerable, covering 19 videos). The balanced-inference experiment prompt was used; the selected evidence snippet was supplied in advance. The exact-refusal proxy classified 4/10 labeled answerable snippets and 2/10 labeled unanswerable snippets as positive (TP 4, FP 2, FN 6, TN 8; proxy precision 0.667, recall 0.400, F1 0.500). This is a small balanced development sample and does not measure whether answers are correct or grounded. An answer with a differently worded refusal can also be counted as positive.

A follow-up inspection of those two public negative cases found responses that appeared to infer substantive details not present in the provided excerpts; one response was incomplete. This is a qualitative inspection, not independent human annotation, and the full source videos were not reviewed. It identifies a plausible unsupported-answer failure mode rather than estimating how often it occurs. The experimental prompt has not replaced the default. Course-derived examples, annotations and results remain private.

The recorded pilot used the earlier evaluator, which could count incomplete responses as proxy-positive. The evaluator now aborts without saving an aggregate if the model reports that an experimental response was cut off; the earlier pilot's counts retain this limitation.
