# PopQA Official Self-RAG Reproduction

Clean-from-zero reproduction of the PopQA short-form QA setup from [`akariasai/self-rag`](https://github.com/akariasai/self-rag), focused on the PopQA result reported in the Self-RAG paper.

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

## Differences From Upstream Self-RAG

The experiment is intended to follow the original Self-RAG PopQA pipeline. The following local changes were made only to make the official code runnable and to keep the repository lightweight:

| Area | Difference | Why |
|---|---|---|
| `official_self_rag/self-rag/retrieval_lm/run_short_form.py` | Added `max_depth=None` to the `call_model_rerank_w_scores_batch(...)` function signature. | The official script later calls this helper with `max_depth=args.max_depth`; without accepting that keyword, the run crashes with `TypeError`. The argument is unused, so this does not change scoring or generation behavior. |
| Repository contents | Model caches, Hugging Face caches, Python bytecode, and `*_tmp` checkpoint files are excluded. | These files are local runtime artifacts and are too large/noisy for GitHub. |
| README/result packaging | Added this README and saved the completed PopQA output JSON. | This documents the exact reproduction command, result, and environment. |
| Environment | Used a Python 3.10 conda environment with CUDA-compatible versions of the official dependencies. | The container's default Python was too new for the original `vllm==0.2.6` / `torch==2.1.2` stack. |

No topic partitioning, CRAG logic, local judge, or custom RAG pipeline is used in this repository. The reported result comes from the official Self-RAG short-form evaluation script with adaptive retrieval.

## Environment

The official dependency stack was installed in a Python 3.10 conda environment:

- `torch==2.1.2+cu121`
- `vllm==0.2.6`
- `flash-attn==2.3.6`
- `transformers==4.36.2`

Model cache files are intentionally not tracked.
