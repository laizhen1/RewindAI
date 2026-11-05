# Experimental local answer evaluation

The optional answer generator is a research prototype. Retrieval can return irrelevant excerpts and a model can invent or distort details even when it says information is missing. Neither a fluent answer nor a timestamp proves correctness.

We ran Qwen2.5-3B-Instruct Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe` on CPU with `llama-cpp-python` 0.3.36, keyword retrieval, temperature 0, and up to 160 output tokens. On 12 short hand-written synthetic cases, it answered six supported questions and declined six unsupported questions; the 0.5B model invented answers on five of the six unsupported cases. The examples were designed for development and are not a benchmark.

We then ran the actual transcript pipeline on nine existing draft questions for the public [KI-Campus ethics video](PUBLIC_SAMPLE_ZH.md), using locally generated Whisper base subtitles. Three draft questions had reference answers and six were intended as unanswerable. Labels came from captions and have not been independently reviewed against the video. In one failure, asked which diabetes treatment the AI recommended, the model said it recommended a diabetes treatment even though the retrieved text did not mention diabetes, then said the specific treatment was not provided. The generated caveat did not undo the unsupported claim. This example illustrates a limitation, not a prevalence estimate. No numerical answer accuracy or F1 is claimed.

A follow-up on the same nine draft questions used a stricter prompt that explicitly treats question assumptions as unverified and requests a fixed insufficient-evidence response when details are missing. In this run, the diabetes-treatment question received that insufficient-evidence response instead of the earlier unsupported claim; the other two related unanswerable questions that reached the model also received it. The three unrelated questions were rejected by retrieval before generation. The three draft answerable questions still received answers. This is a previously inspected nine-question development sample, not an independent test or evidence of a general reliability improvement. Prompt wording does not prevent hallucinations or prompt injection. The pipeline's `abstained` flag records retrieval abstention only; it does not classify whether generated text declines to answer.

To create a local draft for human review, run:

```bash
python scripts/evaluate_local_answers.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --questions PATH_TO_QUESTIONS_JSON --model PATH_TO_GGUF --output artifacts/local_answer_review/run.json
```

The question JSON uses the same `questions` structure as the prepared temporal question files. The script rejects test-split files, fingerprints inputs before and after execution, and writes only to a new JSON file under `artifacts/` or `local_data/`. A source-bound question file is checked against the supplied video and subtitles; unbound files are marked as such. Results include question text, draft answers, retrieved excerpts, timings, input fingerprints and the recorded generation configuration. Keep private course outputs local. Human answer and evidence verdicts start empty; retrieval abstention is not a judgment that a generated answer is correct. Review the source recording, references and each claim before calculating or presenting metrics.
## Start from local question labels

For labels created in the local temporal-label page, the helper below verifies that the CSV is bound to the same video and that its split is development or validation. It creates answer drafts using CPU inference, then starts the review page. The final test split is reserved. All inputs and outputs stay local.

```bash
python scripts/start_answer_review.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --labels artifacts/private_lecture_review/labels.csv --video-id VIDEO_ID --split development --model PATH_TO_GGUF
```

Open `http://127.0.0.1:5097/` after processing. Draft questions must exist first. The model may make unsupported claims; a draft and its resulting scores are not independently verified ground truth. Source fingerprints, questions, answers and reports can be private.

## Review without editing JSON

Start the local review page with the same source video used to produce the review draft:

```bash
python scripts/review_answers.py --video PATH_TO_VIDEO --review artifacts/local_answer_review/run.json
```

Open `http://127.0.0.1:5097/` on the same computer. Play the reference and candidate timestamps, check the question and reference, and enter an answer and evidence verdict for every row. If a reference is wrong, correct the original question labels and regenerate the draft instead of assigning a misleading score. After reviewing all rows, declare the review and save. A new `reviewed-*.json` file is created beside the input; the original is untouched. The page displays question-level metrics from the declared judgments. These are private draft results, and neither the judgments nor the reference labels are independently verified. Changes in the source video or report block saving.

## Human review and question-level metrics

After creating a review draft, a person must check every reference and generated answer against the recording, including whether the cited evidence supports every claim. Alternatively, in the local JSON, set `human_answer_verdict` to `correct`, `incorrect` or `abstain`, and `human_evidence_verdict` to `supported`, `unsupported` or `not_applicable`. Use `correct` only for an answerable question when both the answer and evidence are supported. If the answer adds an unsupported claim, mark it incorrect even if it contains a caveat. Use `abstain` only if the answer makes no unsupported answer claim, and pair it with `not_applicable`. The `retrieval_abstained` field only describes retrieval and must not automatically determine the judgment.

Once every row and reference has actually been reviewed, save the edited review locally and run:

```bash
python scripts/score_answer_review.py --review artifacts/local_answer_review/run.json --output artifacts/local_answer_review/summary.json --confirm-human-review
```

The flag is a declaration by the reviewer, not proof of independent review. Each correct supported answer is a true positive; an incorrect answer is a false positive and, for an answerable question, also a false negative. An abstention on an answerable question is a false negative; one on an unanswerable question is a true negative. Precision, recall and F1 are micro-averaged across questions; undefined denominators produce `null`. The new summary is saved locally without questions or answers, but its hash and metrics can still be private. The script rejects incomplete or inconsistent judgments and reserved test splits. It does not verify the human judgments or turn draft references into a gold standard.

## Lexical answer overlap is a separate diagnostic

The legacy `scripts/evaluate_answers.py` compares generated answer strings against nonempty reference strings. English-like words are treated as word tokens, while CJK ideographs are treated individually; it reports normalized exact match and mean lexical token F1. It cannot assess whether a paraphrase is correct, whether evidence supports the answer, or whether an answer should have been refused. Empty references are rejected; evaluate unanswerable questions separately. It also rejects duplicate question IDs. Its tokenization changed from the earlier character-level implementation, so scores from the two versions must not be compared directly. Private inputs and resulting metrics must stay local.
## Public annotated snippet experiment (development only)

We selected 25 script-answerable and 25 unanswerable examples from the previously examined YTCommentQA video-hash partition, ranking row IDs by SHA-256 within each class. The 50 examples represent 48 videos; visual-only and combined-required labels were excluded. The dataset's human labels concern annotated snippets, not whole-video answerability. The input to the workspace retrieval was the caption text within each question's annotated evidence range, which supplies the relevant region in advance and makes the task easier than whole-video retrieval. No actual source videos or audio were reviewed. This class-balanced subset and previously explored partition are not an independent final test.

Using keyword retrieval and local Qwen2.5 3B Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe` on CPU, prompt `explicit-evidence-v2`, temperature 0 and 160 output tokens, we compared two **answerability proxies**:

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
