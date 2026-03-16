"""
Standalone runner for the LatentMAS-DD method (HuggingFace only, no vLLM).

Usage example:
    python run.py --model_name Qwen/Qwen3-4B --task medqa --prompt sequential
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.latent_mas_dd import LatentMASDDMethod
from methods.latent_mas_hybrid import LatentMASHybridMethod
from methods.text_mas import TextMASMethod
from train_alignment import get_default_alignment_path, train_dd_alignment
from models import ModelWrapper
from utils import auto_device, set_seed


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def load_checkpoint(output_file: str) -> List[Dict]:
    if not output_file or not os.path.exists(output_file):
        return []
    preds = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    preds.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return preds


def append_to_jsonl(output_file: str, results: List[Dict]) -> None:
    if not output_file:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for res in results:
            res.pop("agents", None)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


def auto_output_file(args: argparse.Namespace) -> str:
    model_short = args.model_name.split("/")[-1].lower()
    return os.path.join(
        "results",
        f"{model_short}_{args.task}_{args.method}_{args.prompt}_seed{args.seed}.jsonl",
    )


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
    output_file: str = "",
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print(f"[Latent Decoded] ({latent_steps} steps)")
            else:
                print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    append_to_jsonl(output_file, results)
    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser(description="LatentMAS-DD standalone runner")

    # core args
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "latent_mas_dd", "latent_mas_hybrid"], default="latent_mas_dd")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--task", choices=[
        "gsm8k", "aime2024", "aime2025", "gpqa",
        "arc_easy", "arc_challenge", "mbppplus", "humanevalplus", "medqa", "custom",
    ], default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--custom_prompt_file", type=str, default=None)
    parser.add_argument("--custom_question", type=str, default=None)
    parser.add_argument("--custom_question_file", type=str, default=None)
    parser.add_argument("--custom_gold", type=str, default=None)

    # model / generation args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20)
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", nargs="?", const="<think>\n", default=None)
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--first_agent_text", action="store_true")
    parser.add_argument("--do_not_enforce_qwen", action="store_true")
    parser.add_argument("--agent_models", nargs="+", default=None, help="Per-agent model names for hybrid method (one per agent)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    # ---- Parse custom prompts ----
    args.custom_prompts = None
    args.custom_prompt_text = None
    args.custom_agents = None
    if args.custom_prompt_file:
        with open(args.custom_prompt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        if not raw_text.strip():
            raise ValueError("Custom prompt file is empty.")
        try:
            parsed = json.loads(raw_text)
            args.custom_prompts = parsed
            if isinstance(parsed, dict):
                args.custom_prompt_text = parsed.get("baseline") or parsed.get("user")
                if "agents" in parsed and isinstance(parsed["agents"], list):
                    from methods import Agent
                    args.custom_agents = [
                        Agent(name=ad.get("name", ""), role=ad.get("role", ""))
                        for ad in parsed["agents"]
                        if isinstance(ad, dict) and "name" in ad and "role" in ad
                    ]
        except json.JSONDecodeError:
            args.custom_prompts = raw_text
            args.custom_prompt_text = raw_text

    if not args.output_file:
        args.output_file = auto_output_file(args)

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)

    start_time = time.time()

    # ---- Build method ----
    common_kwargs = dict(temperature=args.temperature, top_p=args.top_p)

    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas":
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas_hybrid":
        method = LatentMASHybridMethod(
            model,
            agent_models=args.agent_models,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas_dd":
        alignment_path = get_default_alignment_path(args.model_name)
        if not os.path.exists(alignment_path):
            print(f"[LatentMAS-DD] Alignment matrix not found at {alignment_path}. Training ...")
            dataset_dir = {
                "gsm8k": {"path": "gsm8k", "name": "main"},
                "aime2024": "HuggingFaceH4/aime_2024",
                "aime2025": "yentinglin/aime_2025",
                "gpqa": "fingertap/GPQA-Diamond",
                "arc_easy": {"path": "allenai/ai2_arc", "name": "ARC-Easy"},
                "arc_challenge": {"path": "allenai/ai2_arc", "name": "ARC-Challenge"},
                "mbppplus": "evalplus/mbppplus",
                "humanevalplus": "evalplus/humanevalplus",
                "medqa": {"path": "json", "data_files": "./data/medqa.json"},
            }
            train_dd_alignment(
                model.model, model.tokenizer, alignment_path,
                device=str(next(model.model.parameters()).device),
                seed=args.seed,
                data_set=dataset_dir.get(args.task, {"path": "gsm8k", "name": "main"}),
            )

        method = LatentMASDDMethod(
            model,
            alignment_path=alignment_path,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )

    # ---- Load dataset ----
    preds: List[Dict] = []
    resumed = 0
    if args.resume:
        preds = load_checkpoint(args.output_file)
        resumed = len(preds)
        if resumed > 0:
            print(f"[Resume] Resuming from {args.output_file}, already processed {resumed} samples.")
    processed = resumed

    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split="train")
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split="test")
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split="test")
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split="test")
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split="test")
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split="test")
    elif args.task == "medqa":
        dataset_iter = load_medqa(split="test")
    elif args.task == "custom":
        if args.custom_question is None and args.custom_question_file is None:
            raise ValueError("For --task custom, provide --custom_question or --custom_question_file.")
        if args.custom_question_file:
            with open(args.custom_question_file, "r", encoding="utf-8") as f:
                custom_question = f.read()
        else:
            custom_question = args.custom_question
        gold = args.custom_gold.strip().lower() if args.custom_gold else ""
        dataset_iter = [{"question": custom_question.strip(), "solution": gold, "gold": gold}]
        if args.max_samples == -1:
            args.max_samples = 1
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)
        args.max_samples = len(dataset_iter)

    if resumed > 0 and resumed >= args.max_samples:
        print(f"[Done] Already processed {resumed} samples >= max_samples={args.max_samples}.")
        progress = tqdm(total=args.max_samples, initial=args.max_samples)
        progress.close()
    else:
        dataset_iter = list(dataset_iter)[resumed:]
        progress = tqdm(total=args.max_samples, initial=resumed)
        batch: List[Dict] = []

        for item in dataset_iter:
            if processed >= args.max_samples:
                break
            batch.append(item)
            if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
                processed, preds = process_batch(
                    method, batch, processed, preds, progress,
                    args.max_samples, args, output_file=args.output_file,
                )
                batch = []
                if processed >= args.max_samples:
                    break

        if batch and processed < args.max_samples:
            processed, preds = process_batch(
                method, batch, processed, preds, progress,
                max_samples=args.max_samples, args=args, output_file=args.output_file,
            )
        progress.close()

    total_time = time.time() - start_time
    acc, correct = evaluate(preds)

    summary = {
        "method": args.method,
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / args.max_samples, 4) if args.max_samples > 0 else 0,
        "output_file": args.output_file,
    }

    summary_file = os.path.splitext(args.output_file)[0] + "_summary.json"
    os.makedirs(os.path.dirname(os.path.abspath(summary_file)), exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Done] Evaluation completed in {round(total_time, 2)} seconds. Summary saved to {summary_file}.")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
