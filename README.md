# PopQA Official Self-RAG Reproduction

Clean-from-zero reproduction of the PopQA short-form QA setup from `akariasai/self-rag`.

## Official Target

The official Self-RAG repo evaluates PopQA with:

```bash
python run_short_form.py   --model_name selfrag/selfrag_llama2_7b   --input_file eval_data/popqa_longtail_w_gs.jsonl   --mode adaptive_retrieval   --max_new_tokens 100   --threshold 0.2   --output_file results/popqa_selfrag_7b_adaptive.json   --metric match   --ndocs 10   --use_groundness   --use_utility   --use_seqscore   --dtype half
```

Paper target:

| Model | PopQA Acc |
|---|---:|
| Self-RAG 7B | 54.9 |
| Self-RAG 13B | 55.8 |

## Reproduced Result

This reproduction uses the official pre-retrieved PopQA long-tail file and the official `selfrag/selfrag_llama2_7b` checkpoint.

| Run | Examples | Metric | Result |
|---|---:|---|---:|
| Self-RAG 7B adaptive retrieval | 1,399 | match / accuracy | 54.97 |

The saved result is:

`results/popqa_selfrag_7b_adaptive_full.json`

The exact JSON value is:

```text
0.5496783416726233
```

This matches the paper's reported Self-RAG 7B PopQA result of `54.9`.

## Environment

The official dependency stack was installed in a Python 3.10 conda environment:

- `torch==2.1.2+cu121`
- `vllm==0.2.6`
- `flash-attn==2.3.6`
- `transformers==4.36.2`

One upstream compatibility patch was required in the official runner: `retrieval_lm/run_short_form.py` now accepts the unused `max_depth` keyword that the same file passes internally during generation.

Model cache files are intentionally not tracked.
