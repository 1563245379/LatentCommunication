from typing import Dict, List, Optional, Tuple

from methods import sequential_default_agents, hierarchical_default_agents
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_message_hybrid_latent_mas,
    LATENT_PLACEHOLDER,
)
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
from tqdm import tqdm

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def transfer_via_realignment(
    hidden_states: torch.Tensor,
    model_from: ModelWrapper,
    model_to: ModelWrapper,
    lambda_reg: float = 1e-5
) -> torch.Tensor:
    """
    Transfer hidden states using the pre-trained data-driven alignment matrix
    stored in ``model_from._latent_realign_matrices``.

    This reuses the same matrix that ``_apply_latent_realignment`` uses
    (trained by ``train_alignment.py`` / loaded by ``LatentMASDDMethod``),
    so cross-model embedding weight computation is no longer needed.
    """
    batch_size, seq_len, dim = hidden_states.shape
    original_dtype = hidden_states.dtype

    matrix, target_norm = model_from._ensure_latent_realign_matrix(
        model_from.model, hidden_states.device, model_from.args
    )

    hidden_flat = hidden_states.reshape(-1, dim).float()
    aligned_flat = torch.matmul(hidden_flat, matrix)  # [batch*seq, dim]
    aligned_norm = aligned_flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    aligned_flat = aligned_flat * (target_norm / aligned_norm)

    return aligned_flat.reshape(batch_size, seq_len, -1).to(original_dtype)


class LatentMASHybridMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        agent_models: Optional[List[str]] = None,
        alignment_path: str = "",
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.initial_model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = getattr(args, "custom_agents", None) or (sequential_default_agents() if args.prompt == "sequential" else hierarchical_default_agents())
        self.method_name = 'latent_mas_hybrid'
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        self.first_agent_text = bool(getattr(args, "first_agent_text", False)) if args else False
        self.task = args.task

        if self.latent_only:
            self.sequential_info_only = True

        # Agent-to-model mapping
        if agent_models is None:
            self.agent_models = [model.model_name] * len(self.agents)
        else:
            assert len(agent_models) == len(self.agents), "Must specify model for each agent"
            self.agent_models = agent_models

        # Load all unique models
        self.models: Dict[str, ModelWrapper] = {model.model_name: model}
        self._load_additional_models()

        # Load DD alignment matrix for each model
        if alignment_path:
            self._load_dd_alignment(model, alignment_path)

        self.model = model

    def _load_additional_models(self):
        """Load any models needed by agents that aren't already loaded."""
        unique_models = set(self.agent_models)
        for model_name in unique_models:
            if model_name not in self.models:
                print(f"Loading additional model: {model_name}")
                new_model = ModelWrapper(
                    model_name,
                    self.initial_model.device,
                    args=self.args,
                )
                self.models[model_name] = new_model

    @staticmethod
    def _load_dd_alignment(model: ModelWrapper, alignment_path: str):
        """Load pre-trained DD alignment matrix and override ModelWrapper's matrix."""
        checkpoint = torch.load(alignment_path, map_location="cpu", weights_only=True)
        W_align = checkpoint["W_align"].float()
        target_norm = checkpoint["target_norm"].float()

        target_device = model.device
        W_align = W_align.to(target_device)
        target_norm = target_norm.to(target_device)

        key = id(model.model)
        model._latent_realign_matrices[key] = (W_align, target_norm)

        print(f"[HYBRID] Loaded DD alignment matrix from {alignment_path}")
        print(f"  Matrix shape: {W_align.shape}, Target norm: {target_norm.item():.4f}")

    def _capture_hidden_states_from_model(
        self,
        agent_model: ModelWrapper,
        wrapped_ids: Optional[torch.Tensor],
        wrapped_mask: torch.Tensor,
        past_kv: Optional[Tuple],
        latent_steps: int,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple, torch.Tensor]:
        """
        Run latent generation and capture RAW hidden states.

        If *inputs_embeds* is provided (soft-token mode), it is used for the
        initial forward pass instead of *wrapped_ids*.  In this mode *past_kv*
        is ignored.
        """
        attention_mask = wrapped_mask.to(agent_model.device)

        if inputs_embeds is not None:
            outputs = agent_model.model(
                inputs_embeds=inputs_embeds.to(agent_model.device),
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            input_ids = wrapped_ids.to(agent_model.device)
            if past_kv is not None:
                past_len = _past_length(past_kv)
                if past_len > 0:
                    past_mask = torch.ones(
                        (attention_mask.shape[0], past_len),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
            outputs = agent_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        raw_latent_hidden_list = []
        for _ in range(latent_steps):
            raw_latent_hidden_list.append(last_hidden.unsqueeze(1))

            latent_vec = agent_model._apply_latent_realignment(last_hidden, agent_model.model)
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )

            outputs = agent_model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        if latent_steps > 0:
            raw_latent_hidden_states = torch.cat(raw_latent_hidden_list, dim=1)
        else:
            batch_size = attention_mask.shape[0]
            hidden_dim = last_hidden.shape[-1]
            raw_latent_hidden_states = torch.zeros(
                (batch_size, 0, hidden_dim), device=last_hidden.device, dtype=last_hidden.dtype
            )

        return past, raw_latent_hidden_states

    def _build_soft_token_embeds(
        self,
        prompts: List[str],
        agent_model: ModelWrapper,
        latent_hiddens: torch.Tensor,
        source_model: ModelWrapper,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split *prompts* at ``LATENT_PLACEHOLDER`` and splice in soft tokens.

        Returns ``(combined_embeds, combined_mask)`` with shapes
        ``[B, total_len, dim]`` and ``[B, total_len]``.
        """
        batch_size = len(prompts)
        embed_layer = agent_model.model.get_input_embeddings()
        soft_tokens = transfer_via_realignment(latent_hiddens, source_model, agent_model)

        combined_embeds_list: List[torch.Tensor] = []
        for i in range(batch_size):
            prompt = prompts[i]
            if LATENT_PLACEHOLDER in prompt:
                before, after = prompt.split(LATENT_PLACEHOLDER, 1)
            else:
                before, after = prompt, ""

            parts: List[torch.Tensor] = []
            if before:
                before_ids = agent_model.tokenizer(
                    before, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(agent_model.device)
                parts.append(embed_layer(before_ids).squeeze(0))

            parts.append(soft_tokens[i])  # [n_soft, dim]

            if after:
                after_ids = agent_model.tokenizer(
                    after, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(agent_model.device)
                parts.append(embed_layer(after_ids).squeeze(0))

            combined_embeds_list.append(torch.cat(parts, dim=0))  # [seq_i, dim]

        max_len = max(e.shape[0] for e in combined_embeds_list)
        dim = combined_embeds_list[0].shape[1]
        dtype = combined_embeds_list[0].dtype

        padded_embeds = torch.zeros(
            batch_size, max_len, dim, dtype=dtype, device=agent_model.device
        )
        padded_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=agent_model.device
        )
        for i in range(batch_size):
            seq_len = combined_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = combined_embeds_list[i]
            padded_mask[i, :seq_len] = 1

        return padded_embeds, padded_mask

    def _generate_from_embeds(
        self,
        agent_model: ModelWrapper,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        """Generate text with *inputs_embeds* as the full prompt (soft tokens included)."""
        input_len = inputs_embeds.shape[1]
        outputs = agent_model.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=agent_model.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx in range(inputs_embeds.shape[0]):
            generated_ids = sequences[idx, input_len:]
            text = agent_model.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            generations.append(text)
        return generations

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        current_model_name: Optional[str] = None

        cumulative_latent_hiddens: Optional[torch.Tensor] = None

        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        agent_pbar = tqdm(self.agents, desc="Agents", unit="agent")
        for agent_idx, agent in enumerate(agent_pbar):
            agent_pbar.set_description(f"Agent: {agent.name} ({agent.role})")
            is_first_agent = (agent_idx == 0)
            is_last_agent = (agent_idx == len(self.agents) - 1)
            should_generate_text = is_first_agent and self.first_agent_text

            agent_model_name = self.agent_models[agent_idx]
            agent_model = self.models[agent_model_name]

            model_switched = (current_model_name is not None and
                              agent_model_name != current_model_name)

            if model_switched:
                print(f"\n[HYBRID] Model switch: {current_model_name} -> {agent_model_name}")
                print(f"[HYBRID] Cumulative latent hiddens shape: "
                      f"{cumulative_latent_hiddens.shape if cumulative_latent_hiddens is not None else None}")

            has_latent = cumulative_latent_hiddens is not None

            batch_messages = [
                build_agent_message_hybrid_latent_mas(
                    role=agent.role, question=item["question"],
                    has_latent=has_latent, args=self.args)
                for item in items
            ]
            prompts = [agent_model.render_chat(msgs) for msgs in batch_messages]

            if not is_last_agent:
                if should_generate_text:
                    if self.args.think:
                        first_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                    else:
                        first_agent_prompts = prompts

                    first_encoded = agent_model.tokenizer(
                        first_agent_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    first_ids = first_encoded["input_ids"].to(agent_model.device)
                    first_mask = first_encoded["attention_mask"].to(agent_model.device)
                    first_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(first_ids, first_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        first_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                    generated_batch, _ = agent_model.generate_text_batch(
                        first_ids,
                        first_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=None,
                    )

                    if current_model_name is None:
                        current_model_name = agent_model_name

                    for idx in range(batch_size):
                        text_out = generated_batch[idx].strip()
                        mask = first_mask[idx].bool()
                        trimmed_ids = first_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append({
                            "name": agent.name,
                            "role": agent.role,
                            "input": first_agent_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": first_tokens_batch[idx],
                            "output": text_out,
                        })
                    continue

                # Latent generation: soft-token insertion when prior latent exists
                if self.args.think:
                    wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                if has_latent:
                    print(f"\n[HYBRID] Injecting {cumulative_latent_hiddens.shape[1]} soft tokens for agent '{agent.name}'")
                    source_model = self.models[current_model_name]
                    combined_embeds, combined_mask = self._build_soft_token_embeds(
                        wrapped_prompts, agent_model,
                        cumulative_latent_hiddens, source_model,
                    )
                    _, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model, None, combined_mask, None, self.latent_steps,
                        inputs_embeds=combined_embeds,
                    )
                    n_soft = cumulative_latent_hiddens.shape[1]
                else:
                    wrapped_encoded = agent_model.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(agent_model.device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(agent_model.device)
                    _, new_latent_hiddens = self._capture_hidden_states_from_model(
                        agent_model, wrapped_ids, wrapped_mask, None, self.latent_steps,
                    )
                    n_soft = 0

                if cumulative_latent_hiddens is None:
                    cumulative_latent_hiddens = new_latent_hiddens
                else:
                    cumulative_latent_hiddens = torch.cat(
                        [cumulative_latent_hiddens, new_latent_hiddens], dim=1
                    )
                current_model_name = agent_model_name

                for idx in range(batch_size):
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "soft_tokens_injected": n_soft,
                        "latent_steps": self.latent_steps,
                        "output": "",
                    })
            else:
                # Last agent: inject soft tokens within prompt, then generate text
                if self.args.think:
                    final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    final_agent_prompts = prompts

                if has_latent:
                    print(f"\n[HYBRID] Injecting {cumulative_latent_hiddens.shape[1]} soft tokens for agent '{agent.name}'")
                    source_model = self.models[current_model_name]
                    combined_embeds, combined_mask = self._build_soft_token_embeds(
                        final_agent_prompts, agent_model,
                        cumulative_latent_hiddens, source_model,
                    )
                    generated_batch = self._generate_from_embeds(
                        agent_model, combined_embeds, combined_mask,
                        self.judger_max_new_tokens, self.temperature, self.top_p,
                    )
                    n_soft = cumulative_latent_hiddens.shape[1]
                else:
                    final_encoded = agent_model.tokenizer(
                        final_agent_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    final_ids = final_encoded["input_ids"].to(agent_model.device)
                    final_mask = final_encoded["attention_mask"].to(agent_model.device)
                    generated_batch, _ = agent_model.generate_text_batch(
                        final_ids,
                        final_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    n_soft = 0

                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": final_agent_prompts[idx],
                        "soft_tokens_injected": n_soft,
                        "output": final_text,
                    })

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")
                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    if pred is None or pred == "":
                        ok = False
                    else:
                        ok = (int(pred) == int(gold))
                except ValueError:
                    ok = False
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False

            results.append({
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
            })
        return results
