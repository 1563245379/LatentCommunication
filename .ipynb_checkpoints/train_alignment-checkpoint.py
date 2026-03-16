"""
Train a data-driven alignment matrix for LatentMAS-DD method.

Uses ridge regression to learn a linear mapping from the last hidden layer
to the target layer (default: layer 0, i.e. input embeddings) with temporal
shift (h_L[t] -> h_K[t+1]).

Can be run standalone:
    python train_alignment.py --model_name Qwen/Qwen3-4B --n_train 500

Or imported from run.py for automatic training when weights are missing.
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm


def get_default_alignment_path(model_name, target_layer_idx=0):
    model_short = model_name.split("/")[-1].lower()
    return os.path.join("weights", f"{model_short}_dd_alignment_layer{target_layer_idx}.pt")


@torch.no_grad()
def collect_shifted_hidden_pairs(model, tokenizer, questions, target_layer_idx, device):
    """Collect (h_last[t], h_target[t+1]) pairs via temporal shift."""
    all_source, all_target = [], []

    for q in tqdm(questions, desc="Collecting hidden state pairs"):
        inputs = tokenizer(q, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)

        h_last = outputs.hidden_states[-1][0].cpu().float()
        h_target = outputs.hidden_states[target_layer_idx][0].cpu().float()

        if h_last.shape[0] < 2:
            continue

        # Temporal shift: source t -> target t+1
        all_source.append(h_last[:-1])
        all_target.append(h_target[1:])

    return torch.cat(all_source, dim=0), torch.cat(all_target, dim=0)


def train_ridge_regression(source, target, lambda_reg=1e-4):
    """Solve ridge regression: W = (X^T X + λI)^{-1} X^T Y"""
    d = source.shape[1]
    gram = source.T @ source + lambda_reg * torch.eye(d, device=source.device)
    rhs = source.T @ target
    W_align = torch.linalg.solve(gram, rhs)
    target_norm = target.norm(dim=1).mean()
    return W_align, target_norm


def train_dd_alignment(
    model,
    tokenizer,
    output_path,
    n_train=500,
    target_layer_idx=0,
    lambda_reg=1e-4,
    device="cuda",
    seed=42,
    data_set="gsm8k",
):
    """Train and save a data-driven alignment matrix."""
    from datasets import load_dataset

    np.random.seed(seed)
    torch.manual_seed(seed)
    if isinstance(data_set, dict):
        ds = load_dataset(**data_set, split="train")
    else:
        ds = load_dataset(data_set, split="train")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_train, len(ds)), replace=False)
    questions = [ds[int(i)]["question"].strip() for i in indices]

    print(f"[DD Alignment] Collecting hidden states from {len(questions)} samples (target layer={target_layer_idx}) ...")
    source, target = collect_shifted_hidden_pairs(model, tokenizer, questions, target_layer_idx, device)
    print(f"  Source shape: {source.shape}, Target shape: {target.shape}")

    print(f"[DD Alignment] Training ridge regression (λ={lambda_reg}) ...")
    W_align, target_norm = train_ridge_regression(source, target, lambda_reg)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(
        {
            "W_align": W_align,
            "target_norm": target_norm,
            "target_layer_idx": target_layer_idx,
            "lambda_reg": lambda_reg,
            "n_train": n_train,
        },
        output_path,
    )
    print(f"[DD Alignment] Saved alignment matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train data-driven alignment matrix for LatentMAS-DD")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--target_layer_idx", type=int, default=0)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if not args.output:
        args.output = get_default_alignment_path(args.model_name, args.target_layer_idx)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
    )
    model.eval()

    train_dd_alignment(
        model, tokenizer, args.output,
        n_train=args.n_train,
        target_layer_idx=args.target_layer_idx,
        lambda_reg=args.lambda_reg,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
