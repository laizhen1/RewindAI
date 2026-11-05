# Experimental local answer evaluation

The optional answer generator is a research prototype. Retrieval can return irrelevant excerpts and a model can invent or distort details even when it says information is missing. Neither a fluent answer nor a timestamp proves correctness.

We ran Qwen2.5-3B-Instruct Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe` on CPU with `llama-cpp-python` 0.3.36, keyword retrieval, temperature 0, and up to 160 output tokens. On 12 short hand-written synthetic cases, it answered six supported questions and declined six unsupported questions; the 0.5B model invented answers on five of the six unsupported cases. The examples were designed for development and are not a benchmark.

We then ran the actual transcript pipeline on nine existing draft questions for the public [KI-Campus ethics video](PUBLIC_SAMPLE_ZH.md), using locally generated Whisper base subtitles. Three draft questions had reference answers and six were intended as unanswerable. Labels came from captions and have not been independently reviewed against the video. In one failure, asked which diabetes treatment the AI recommended, the model said it recommended a diabetes treatment even though the retrieved text did not mention diabetes, then said the specific treatment was not provided. The generated caveat did not undo the unsupported claim. This example illustrates a limitation, not a prevalence estimate. No numerical answer accuracy or F1 is claimed.

To create a local draft for human review, run:

```bash
python scripts/evaluate_local_answers.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --questions PATH_TO_QUESTIONS_JSON --model PATH_TO_GGUF --output artifacts/local_answer_review/run.json
```

The question JSON uses the same `questions` structure as the prepared temporal question files. The script rejects test-split files, fingerprints inputs before and after execution, and writes only to a new JSON file under `artifacts/` or `local_data/`. A source-bound question file is checked against the supplied video and subtitles; unbound files are marked as such. Results include question text, draft answers, retrieved excerpts, timings and input fingerprints. Keep private course outputs local. Human answer and evidence verdicts start empty; retrieval abstention is not a judgment that a generated answer is correct. Review the source recording, references and each claim before calculating or presenting metrics.
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
