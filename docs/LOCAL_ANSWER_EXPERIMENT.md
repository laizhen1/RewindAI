# Experimental local answer evaluation

The optional answer generator is a research prototype. Retrieval can return irrelevant excerpts and a model can invent or distort details even when it says information is missing. Neither a fluent answer nor a timestamp proves correctness.

We ran Qwen2.5-3B-Instruct Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe` on CPU with `llama-cpp-python` 0.3.36, keyword retrieval, temperature 0, and up to 160 output tokens. On 12 short hand-written synthetic cases, it answered six supported questions and declined six unsupported questions; the 0.5B model invented answers on five of the six unsupported cases. The examples were designed for development and are not a benchmark.

We then ran the actual transcript pipeline on nine existing draft questions for the public [KI-Campus ethics video](PUBLIC_SAMPLE_ZH.md), using locally generated Whisper base subtitles. Three draft questions had reference answers and six were intended as unanswerable. Labels came from captions and have not been independently reviewed against the video. In one failure, asked which diabetes treatment the AI recommended, the model said it recommended a diabetes treatment even though the retrieved text did not mention diabetes, then said the specific treatment was not provided. The generated caveat did not undo the unsupported claim. This example illustrates a limitation, not a prevalence estimate. No numerical answer accuracy or F1 is claimed.

To create a local draft for human review, run:

```bash
python scripts/evaluate_local_answers.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --questions PATH_TO_QUESTIONS_JSON --model PATH_TO_GGUF --output artifacts/local_answer_review/run.json
```

The question JSON uses the same `questions` structure as the prepared temporal question files. The script rejects test-split files, fingerprints inputs before and after execution, and writes only to a new JSON file under `artifacts/` or `local_data/`. A source-bound question file is checked against the supplied video and subtitles; unbound files are marked as such. Results include question text, draft answers, retrieved excerpts, timings and input fingerprints. Keep private course outputs local. Human answer and evidence verdicts start empty; retrieval abstention is not a judgment that a generated answer is correct. Review the source recording, references and each claim before calculating or presenting metrics.
