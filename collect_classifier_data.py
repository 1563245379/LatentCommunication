"""
Collect training data and train the LatentStopClassifier.

For each sample from multiple datasets, runs the latent autoregressive loop
and at each step decodes the hidden state through lm_head. When the EOS
(end-of-sequence) token probability is the highest among all tokens the step
is labelled as "stop" (label=1); otherwise "continue" (label=0).

The collected (hidden_state, label) pairs are then used to train the binary
classifier so it can predict adaptive stopping during inference.

Usage:
    python collect_classifier_data.py \
        --model_name Qwen/Qwen3-4B \
        --max_samples_per_task 200 \
        --latent_steps 20 \
        --output weights/stop_classifier.pt
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import (
    load_gsm8k,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gpqa_diamond,
    load_medqa,
)
from latent_classifier import LatentStopClassifier, save_classifier
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas
from train_alignment import get_default_alignment_path
from utils import auto_device, set_seed


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

AVAILABLE_TASKS: Dict[str, dict] = {
    "gsm8k": {"loader": load_gsm8k, "kwargs": {"split": "train"}},
    "aime2025": {"loader": load_aime2025, "kwargs": {"split": "train"}},
    "arc_easy": {"loader": load_arc_easy, "kwargs": {"split": "train"}},
    "arc_challenge": {"loader": load_arc_challenge, "kwargs": {"split": "train"}},
    "gpqa": {"loader": load_gpqa_diamond, "kwargs": {"split": "train"}},
}


def _load_task_samples(task_name: str, max_samples: int, seed: int) -> List[Dict]:
    """Load up to *max_samples* items from *task_name*."""
    info = AVAILABLE_TASKS[task_name]
    items = list(info["loader"](**info["kwargs"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    return items[:max_samples]


# ---------------------------------------------------------------------------
# Latent-step data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latent_stop_data(
    model: ModelWrapper,
    items: List[Dict],
    *,
    latent_steps: int,
    task: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the latent AR loop and label each step by EOS-argmax criterion.

    Returns:
        hidden_states: [N, hidden_dim]  (float32, CPU)
        labels:        [N]              (float32, CPU)  0=continue, 1=stop
    """
    eos_id = model.tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer does not define an eos_token_id.")

    lm_head = model.model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model.model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot locate lm_head on the model.")

    all_hiddens: List[torch.Tensor] = []
    all_labels: List[int] = []

    for item in tqdm(items, desc=f"Collecting [{task}]"):
        messages = build_agent_message_sequential_latent_mas(
            role="planner",
            question=item["question"],
            context="",
            method="latent_mas_dd",
            args=args,
        )
        prompt_text = model.render_chat(messages, add_generation_prompt=True)
        if args.think:
            prompt_text = f"{prompt_text}{args.think}"

        encoded = model.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        # Initial forward pass
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

        # Autoregressive latent loop
        for step in range(latent_steps):
            # ---- apply realignment ----
            latent_vec = model._apply_latent_realignment(last_hidden, model.model)
            latent_embed = latent_vec.unsqueeze(1)  # [1, 1, hidden_dim]

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (1, past_len + 1), dtype=torch.long, device=model.device,
            )

            outputs = model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

            # ---- decode to check EOS ----
            logits = lm_head(last_hidden)          # [1, vocab_size]
            argmax_id = logits.argmax(dim=-1).item()
            label = 1 if argmax_id == eos_id else 0

            all_hiddens.append(last_hidden.squeeze(0).cpu().float())
            all_labels.append(label)

            # If we already hit stop, still continue collecting remaining
            # steps so the classifier sees both positives and negatives.

    hidden_states = torch.stack(all_hiddens, dim=0)  # [N, hidden_dim]
    labels = torch.tensor(all_labels, dtype=torch.float32)  # [N]
    return hidden_states, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    hidden_dim: int,
    intermediate_dim: int = 256,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> LatentStopClassifier:
    """Train the LatentStopClassifier on collected data."""
    classifier = LatentStopClassifier(hidden_dim, intermediate_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    dataset = TensorDataset(hidden_states, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    print(f"[Train] Samples: {len(labels)} (pos={n_pos}, neg={n_neg})")

    classifier.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_h, batch_l in loader:
            batch_h = batch_h.to(device)
            batch_l = batch_l.to(device)

            preds = classifier(batch_h).squeeze(-1)
            loss = criterion(preds, batch_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_h.size(0)
            correct += ((preds >= 0.5).float() == batch_l).sum().item()
            total += batch_h.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    classifier.eval()
    return classifier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect latent stop data & train LatentStopClassifier",
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(AVAILABLE_TASKS.keys()),
        choices=list(AVAILABLE_TASKS.keys()),
        help="Tasks to collect data from (default: all available).",
    )
    parser.add_argument("--max_samples_per_task", type=int, default=200)
    parser.add_argument("--latent_steps", type=int, default=20,
                        help="Number of latent AR steps per sample for data collection.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--think", nargs="?", const="<think>\n", default=None)
    parser.add_argument("--do_not_enforce_qwen", action="store_true")

    # Training hyper-parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--intermediate_dim", type=int, default=256)

    parser.add_argument("--output", type=str, default="",
                        help="Path to save trained classifier (default: weights/<model>_stop_classifier.pt).")
    parser.add_argument("--save_data", type=str, default="",
                        help="Optionally save raw (hidden_states, labels) for debugging.")
    args = parser.parse_args()

    # Fields expected by prompt builders but not directly relevant here
    args.task = "gsm8k"
    args.prompt = "sequential"
    args.custom_prompts = None
    args.custom_prompt_text = None
    args.custom_agents = None

    set_seed(args.seed)
    device = auto_device(args.device)

    # ---- Load model ----
    print(f"Loading model: {args.model_name} ...")
    model = ModelWrapper(args.model_name, device, args=args)

    # ---- Load data-driven alignment if available ----
    alignment_path = get_default_alignment_path(args.model_name)
    if os.path.exists(alignment_path):
        checkpoint = torch.load(alignment_path, map_location="cpu", weights_only=True)
        W_align = checkpoint["W_align"].float().to(device)
        target_norm = checkpoint["target_norm"].float().to(device)
        key = id(model.model)
        model._latent_realign_matrices[key] = (W_align, target_norm)
        print(f"[Alignment] Loaded data-driven alignment matrix from {alignment_path}")
    elif args.latent_space_realign:
        print("[Alignment] Using parameter-space alignment (no DD matrix found).")

    # ---- Collect data from all tasks ----
    all_hiddens: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for task_name in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        args.task = task_name  # update so prompt builder sees the right task
        items = _load_task_samples(task_name, args.max_samples_per_task, args.seed)
        print(f"  Loaded {len(items)} samples")

        h, l = collect_latent_stop_data(
            model, items,
            latent_steps=args.latent_steps,
            task=task_name,
            args=args,
        )
        all_hiddens.append(h)
        all_labels.append(l)
        n_pos = int(l.sum().item())
        print(f"  Collected {len(l)} vectors (pos={n_pos}, neg={len(l)-n_pos})")

    hidden_states = torch.cat(all_hiddens, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"\n[Total] {len(labels)} vectors "
          f"(pos={int(labels.sum().item())}, neg={int((1-labels).sum().item())})")

    # ---- Optionally save raw data ----
    if args.save_data:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_data)), exist_ok=True)
        torch.save({"hidden_states": hidden_states, "labels": labels}, args.save_data)
        print(f"[Data] Saved raw data to {args.save_data}")

    # ---- Train classifier ----
    hidden_dim = hidden_states.shape[1]
    print(f"\n[Train] hidden_dim={hidden_dim}, intermediate_dim={args.intermediate_dim}")
    classifier = train_classifier(
        hidden_states,
        labels,
        hidden_dim=hidden_dim,
        intermediate_dim=args.intermediate_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # ---- Save classifier ----
    if not args.output:
        model_short = args.model_name.split("/")[-1].lower()
        args.output = os.path.join("weights", f"{model_short}_stop_classifier.pt")
    save_classifier(classifier, args.output, hidden_dim, args.intermediate_dim)


if __name__ == "__main__":
    main()
