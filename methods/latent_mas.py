"""
LatentMAS base method (HuggingFace Transformers only, no vLLM).
"""

from typing import Dict, List, Optional, Tuple

import torch
import argparse
from tqdm import tqdm

from methods import sequential_default_agents, hierarchical_default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = getattr(args, "custom_agents", None) or (sequential_default_agents() if args.prompt == "sequential" else hierarchical_default_agents())
        self.method_name = 'latent_mas'
        self.first_agent_text = bool(getattr(args, "first_agent_text", False)) if args else False
        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            from transformers.cache_utils import DynamicCache
            if isinstance(past_kv, DynamicCache):
                new_cache = DynamicCache()
                for layer_idx, layer in enumerate(past_kv.layers):
                    sliced_k = self._slice_tensor(layer.keys, tokens_to_keep)
                    sliced_v = self._slice_tensor(layer.values, tokens_to_keep)
                    new_cache.update(sliced_k, sliced_v, layer_idx)
                return new_cache
            if hasattr(past_kv, 'to_legacy_cache'):
                legacy = past_kv.to_legacy_cache()
                trimmed_legacy = tuple(
                    tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                    for layer in legacy
                )
                return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    def _filter_latent_past_kv(self, past_kv: Optional[Tuple]) -> Tuple[Optional[Tuple], int]:
        """Hook for subclasses to filter KV cache after latent generation.

        Returns:
            (filtered_past_kv, cache_seq_offset): The filtered cache and a
            position offset that accounts for tokens removed from the front.
        """
        return past_kv, 0

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        cache_seq_offset: int = 0
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        agent_pbar = tqdm(self.agents, desc="Agents", unit="agent")
        for agent_idx, agent in enumerate(agent_pbar):
            agent_pbar.set_description(f"Agent: {agent.name} ({agent.role})")

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role, question=item["question"], context="",
                        method=self.method_name, args=self.args
                    )
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role, question=item["question"], context="",
                        method=self.method_name, args=self.args
                    )
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            is_first_agent = (agent_idx == 0)
            is_last_agent = (agent_idx == len(self.agents) - 1)
            should_generate_text = is_first_agent and self.first_agent_text

            if not is_last_agent:
                if should_generate_text:
                    first_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts] if self.args.think else prompts

                    first_encoded = self.model.tokenizer(
                        first_agent_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    first_ids = first_encoded["input_ids"].to(self.model.device)
                    first_mask = first_encoded["attention_mask"].to(self.model.device)
                    first_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(first_ids, first_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        first_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                    generated_batch, past_kv = self.model.generate_text_batch(
                        first_ids,
                        first_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_kv,
                        cache_seq_offset=cache_seq_offset,
                    )
                    cache_seq_offset = 0  # text generation produces contiguous cache

                    for idx in range(batch_size):
                        text_out = generated_batch[idx].strip()
                        mask = first_mask[idx].bool()
                        trimmed_ids = first_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": first_agent_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": first_tokens_batch[idx],
                                "output": text_out,
                            }
                        )
                else:
                    wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts] if self.args.think else prompts

                    wrapped_encoded = self.model.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                    wrapped_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                    past_kv = self.model.generate_latent_batch(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                        cache_seq_offset=cache_seq_offset,
                    )
                    past_kv, cache_seq_offset = self._filter_latent_past_kv(past_kv)

                    for idx in range(batch_size):
                        mask = wrapped_mask[idx].bool()
                        trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": wrapped_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": wrapped_tokens_batch[idx],
                                "latent_steps": self.latent_steps,
                                "output": "",
                            }
                        )
            else:
                # Last agent: Generate final text output
                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                    final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    final_agent_prompts = prompts

                final_agent_encoded = self.model.tokenizer(
                    final_agent_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                final_agent_ids = final_agent_encoded["input_ids"].to(self.model.device)
                final_agent_mask = final_agent_encoded["attention_mask"].to(self.model.device)
                final_agent_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(final_agent_ids, final_agent_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    final_agent_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    final_agent_ids,
                    final_agent_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                    cache_seq_offset=cache_seq_offset,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = final_agent_mask[idx].bool()
                    trimmed_ids = final_agent_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": final_agent_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": final_agent_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, _ = run_with_timeout(python_code_to_exe, timeout=10)

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    ok = (int(pred) == int(gold))
                except (ValueError, TypeError):
                    ok = False

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
