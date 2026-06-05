#!/usr/bin/env python3
"""Transformers fallback for official Self-RAG short-form PopQA inference.

This mirrors akariasai/self-rag `retrieval_lm/run_short_form.py` for the PopQA
short-form setting while avoiding a vLLM dependency. It keeps the same prompt,
reflection-token logprob scoring, retrieved contexts, and match metric.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO = REPO_ROOT / "experiments/popqa_selfrag_official_reproduction/official_self_rag/self-rag/retrieval_lm"
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from utils import PROMPT_DICT, control_tokens, load_special_tokens  # type: ignore


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def official_match(prediction: str, ground_truth: list[str]) -> int:
    # Official metrics.py uses raw substring matching. We keep that as the main score.
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0


def normalized_match(prediction: str, ground_truth: list[str]) -> int:
    pred = normalize_answer(prediction)
    return int(any(normalize_answer(gt) in pred for gt in ground_truth if normalize_answer(gt)))


def load_jsonlines(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def postprocess_answer(answer: str) -> str:
    for token in control_tokens:
        answer = answer.replace(token, "")
    answer = answer.replace("</s>", "").replace("<|endoftext|>", "")
    answer = answer.replace("\n", "")
    answer = answer.strip()
    if answer.startswith("#") or answer.startswith(":"):
        answer = answer[1:].strip()
    return answer


def preprocess_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["instruction"] = row.get("question", row.get("instruction", ""))
    return rows


def logsumexp(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    m = np.max(arr)
    return float(m + np.log(np.sum(np.exp(arr - m))))


class SelfRagGenerator:
    def __init__(self, model_name: str, cache_dir: str | None, dtype: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side="left")
        torch_dtype = torch.float16 if dtype == "half" else torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = torch.device(device)
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = load_special_tokens(
            self.tokenizer, use_grounding=True, use_utility=True
        )

    @torch.no_grad()
    def generate_with_scores(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        token_ids = [int(x) for x in generated_ids.detach().cpu().tolist()]
        step_logprobs = []
        cumulative = 0.0
        for step_scores, token_id in zip(output.scores, token_ids):
            log_probs = torch.log_softmax(step_scores[0].float(), dim=-1)
            cumulative += float(log_probs[token_id].detach().cpu())
            step_logprobs.append(log_probs.detach().cpu())
        return {"text": text, "token_ids": token_ids, "logprobs": step_logprobs, "cumulative_logprob": cumulative}

    def token_logprob(self, step_logprobs: list[torch.Tensor], step_idx: int, token_id: int) -> float:
        if step_idx >= len(step_logprobs) or token_id is None or token_id < 0:
            return -100.0
        return float(step_logprobs[step_idx][token_id])

    def should_retrieve(self, prompt: str, threshold: float, max_new_tokens: int, mode: str) -> tuple[bool, dict[str, Any] | None]:
        if mode == "always_retrieve":
            return True, None
        if mode == "no_retrieval":
            return False, None
        pred = self.generate_with_scores(prompt, max_new_tokens=max_new_tokens)
        first_step = pred["logprobs"][0] if pred["logprobs"] else None
        if first_step is None:
            return False, pred
        retrieve_lp = self.token_logprob(pred["logprobs"], 0, self.ret_tokens["[Retrieval]"])
        no_retrieve_lp = self.token_logprob(pred["logprobs"], 0, self.ret_tokens["[No Retrieval]"])
        denom = logsumexp([retrieve_lp, no_retrieve_lp])
        retrieve_prob = math.exp(retrieve_lp - denom)
        return retrieve_prob > threshold, {**pred, "retrieval_prob": retrieve_prob}

    def score_retrieval_prediction(self, pred: dict[str, Any], use_seqscore: bool, w_rel: float, w_sup: float, w_use: float) -> dict[str, Any]:
        step_logprobs = pred["logprobs"]
        token_ids = pred["token_ids"]

        rel_values = {tok: math.exp(self.token_logprob(step_logprobs, 0, tid)) for tok, tid in self.rel_tokens.items()}
        rel_sum = sum(rel_values.values()) or 1.0
        relevance_score = rel_values.get("[Relevant]", 0.0) / rel_sum

        ground_score = 0.0
        ground_idx = next((i for i, tok_id in enumerate(token_ids) if tok_id in set(self.grd_tokens.values())), None)
        grd_values = {}
        if ground_idx is not None:
            grd_values = {tok: math.exp(self.token_logprob(step_logprobs, ground_idx, tid)) for tok, tid in self.grd_tokens.items()}
            grd_sum = sum(grd_values.values()) or 1.0
            ground_score = grd_values.get("[Fully supported]", 0.0) / grd_sum + 0.5 * grd_values.get("[Partially supported]", 0.0) / grd_sum

        utility_score = 0.0
        utility_idx = next((i for i, tok_id in enumerate(token_ids) if tok_id in set(self.ut_tokens.values())), None)
        ut_values = {}
        if utility_idx is not None:
            ut_values = {tok: math.exp(self.token_logprob(step_logprobs, utility_idx, tid)) for tok, tid in self.ut_tokens.items()}
            ut_sum = sum(ut_values.values()) or 1.0
            weights = [-1, -0.5, 0, 0.5, 1]
            utility_score = sum(weights[i] * (ut_values.get(f"[Utility:{i + 1}]", 0.0) / ut_sum) for i in range(5))

        seq_score = pred["cumulative_logprob"] / max(len(token_ids), 1)
        final_score = w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
        if use_seqscore:
            final_score += math.exp(seq_score)
        return {
            "final_score": float(final_score),
            "relevance_score": float(relevance_score),
            "ground_score": float(ground_score),
            "utility_score": float(utility_score),
            "seq_score_exp": float(math.exp(seq_score)),
            "rel_values": rel_values,
            "grd_values": grd_values,
            "ut_values": ut_values,
        }

    def answer(self, prompt: str, evidences: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, dict[str, Any], bool]:
        do_retrieve, no_ret_pred = self.should_retrieve(prompt, args.threshold, args.max_new_tokens, args.mode)
        results: dict[str, Any] = {}
        if no_ret_pred is not None:
            results["no_retrieval"] = no_ret_pred["text"]
            if "retrieval_prob" in no_ret_pred:
                results["retrieval_prob"] = no_ret_pred["retrieval_prob"]

        if not do_retrieve:
            pred = self.generate_with_scores(prompt + "[No Retrieval]", args.max_new_tokens)
            return postprocess_answer(pred["text"]), {**results, "final_no_retrieval": pred["text"]}, False

        best_key = None
        best_score = -float("inf")
        for idx, para in enumerate(evidences[: args.ndocs]):
            evidence_prompt = prompt + "[Retrieval]{0}\n{1}".format(para.get("title", ""), para.get("text", ""))
            pred = self.generate_with_scores(evidence_prompt, args.max_new_tokens)
            score = self.score_retrieval_prediction(pred, args.use_seqscore, args.w_rel, args.w_sup, args.w_use)
            key = f"retrieval_{idx}"
            results[key] = {"pred": pred["text"], "score": score["final_score"], "ctx": para, "score_details": score}
            if score["final_score"] > best_score:
                best_score = score["final_score"]
                best_key = key
        if best_key is None:
            return "", results, True
        return results[best_key]["pred"], results, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="selfrag/selfrag_llama2_7b")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--cache_dir", default="experiments/popqa_selfrag_official_reproduction/model_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="half", choices=["half", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--mode", choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"], default="adaptive_retrieval")
    parser.add_argument("--metric", default="match")
    parser.add_argument("--ndocs", type=int, default=10)
    parser.add_argument("--use_groundness", action="store_true")
    parser.add_argument("--use_utility", action="store_true")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_use", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args()

    rows = preprocess_rows(load_jsonlines(args.input_file))
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output_file.with_suffix(args.output_file.suffix + ".tmp")

    start = 0
    final_results = {"preds": [], "prompts": [], "metric_results": [], "normalized_metric_results": [], "all_results": [], "golds": [], "metric": args.metric, "scores": []}
    if args.resume and tmp_path.exists():
        final_results = json.loads(tmp_path.read_text())
        start = len(final_results["preds"])

    generator = SelfRagGenerator(args.model_name, args.cache_dir, args.dtype, args.device)
    retrieve_count = 0
    for row in tqdm(rows[start:], initial=start, total=len(rows)):
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        evidences = row.get("ctxs", row.get("top_contexts", []))[: args.ndocs]
        pred, details, did_retrieve = generator.answer(prompt, evidences, args)
        if did_retrieve:
            retrieve_count += 1
        metric = official_match(pred, row["answers"])
        normalized_metric = normalized_match(pred, row["answers"])
        final_results["preds"].append(pred)
        final_results["prompts"].append(prompt)
        final_results["metric_results"].append(metric)
        final_results["normalized_metric_results"].append(normalized_metric)
        final_results["all_results"].append(details)
        final_results["golds"].append(row["answers"])
        if len(final_results["preds"]) % args.progress_every == 0:
            mean = float(np.mean(final_results["metric_results"]))
            print(f"average: {mean:.4f}", flush=True)
            tmp_path.write_text(json.dumps(final_results), encoding="utf-8")

    final_results["metric_mean"] = float(np.mean(final_results["metric_results"])) if final_results["metric_results"] else 0.0
    final_results["normalized_metric_mean"] = float(np.mean(final_results["normalized_metric_results"])) if final_results["normalized_metric_results"] else 0.0
    final_results["retrieval_frequency"] = retrieve_count / max(len(rows) - start, 1)
    args.output_file.write_text(json.dumps(final_results), encoding="utf-8")
    tmp_path.write_text(json.dumps(final_results), encoding="utf-8")
    print(f"Final result: {final_results['metric_mean']}")
    print(f"Normalized final result: {final_results['normalized_metric_mean']}")
    print(f"Retrieval frequency in this run segment: {final_results['retrieval_frequency']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
