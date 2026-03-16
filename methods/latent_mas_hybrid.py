from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
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
    Transfer hidden states from Model A to Model B using cross-model realignment.

    W_cross = (W_out_A^T @ W_out_A + λI)^-1 @ W_out_A^T @ W_in_B
    embedding_B = hidden_A @ W_cross
    """
    batch_size, seq_len, dim_A = hidden_states.shape
    original_dtype = hidden_states.dtype

    W_out_A = model_from.model.get_output_embeddings().weight  # [vocab_A, dim_A]
    W_in_B = model_to.model.get_input_embeddings().weight      # [vocab_B, dim_B]

    dim_B = W_in_B.shape[1]

    W_out_A_f32 = W_out_A.float()
    W_in_B_f32 = W_in_B.float()

    gram = torch.matmul(W_out_A_f32.T, W_out_A_f32)  # [dim_A, dim_A]
    reg = lambda_reg * torch.eye(gram.shape[0], device=gram.device, dtype=torch.float32)
    gram_reg = gram + reg

    vocab_A, _ = W_out_A.shape
    vocab_B, _ = W_in_B.shape

    if vocab_A != vocab_B:
        min_vocab = min(vocab_A, vocab_B)
        W_out_A_f32 = W_out_A_f32[:min_vocab, :]
        W_in_B_f32 = W_in_B_f32[:min_vocab, :]
        print(f"[WARNING] Vocab size mismatch: {vocab_A} vs {vocab_B}, using first {min_vocab} tokens")

    rhs = torch.matmul(W_out_A_f32.T, W_in_B_f32)  # [dim_A, dim_B]
    W_cross = torch.linalg.solve(gram_reg, rhs)  # [dim_A, dim_B]

    hidden_flat = hidden_states.reshape(-1, dim_A).float()
    embeddings_B_flat = torch.matmul(hidden_flat, W_cross)  # [batch*seq, dim_B]

    target_norm = W_in_B_f32.norm(dim=1).mean()
    current_norms = torch.norm(embeddings_B_flat, dim=1, keepdim=True)
    embeddings_B_flat = embeddings_B_flat * (target_norm / (current_norms + 1e-8))

    embeddings_B = embeddings_B_flat.reshape(batch_size, seq_len, dim_B).to(original_dtype)
    return embeddings_B


class LatentMASHybridMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        agent_models: Optional[List[str]] = None,
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
        self.agents = getattr(args, "custom_agents", None) or default_agents()
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

    def _capture_hidden_states_from_model(
        self,
        agent_model: ModelWrapper,
        wrapped_ids: torch.Tensor,
        wrapped_mask: torch.Tensor,
        past_kv: Optional[Tuple],
        latent_steps: int
    ) -> Tuple[Tuple, torch.Tensor]:
        """
        Run latent generation and capture RAW hidden states (not aligned embeddings).
        """
        input_ids = wrapped_ids.to(agent_model.device)
        attention_mask = wrapped_mask.to(agent_model.device)

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
            batch_size = wrapped_ids.shape[0]
            hidden_dim = last_hidden.shape[-1]
            raw_latent_hidden_states = torch.zeros(
                (batch_size, 0, hidden_dim), device=last_hidden.device, dtype=last_hidden.dtype
            )

        return past, raw_latent_hidden_states

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

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        current_model_name: Optional[str] = None

        cumulative_prompts: List[str] = ["" for _ in range(batch_size)]
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

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role, question=item["question"], context="",
                        method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role, question=item["question"], context="",
                        method=self.method_name, args=self.args)
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = agent_model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

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
                        cumulative_prompts[idx] += first_agent_prompts[idx] + text_out
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

                # Standard latent generation (always via hidden state transfer)
                if self.args.think:
                    wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = agent_model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(agent_model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(agent_model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                # Rebuild context from scratch: re-encode prompts + transfer hidden states
                if cumulative_latent_hiddens is not None:
                    source_model = self.models[current_model_name]

                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)

                    with torch.no_grad():
                        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)

                    transferred_latent_embeds = transfer_via_realignment(
                        cumulative_latent_hiddens, source_model, agent_model
                    )

                    combined_embeds = torch.cat([prompt_embeds, transferred_latent_embeds], dim=1)
                    combined_mask = torch.cat([
                        prompt_mask,
                        torch.ones(
                            (batch_size, transferred_latent_embeds.shape[1]),
                            dtype=prompt_mask.dtype, device=prompt_mask.device)
                    ], dim=1)

                    with torch.no_grad():
                        transfer_outputs = agent_model.model(
                            inputs_embeds=combined_embeds,
                            attention_mask=combined_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True
                        )
                        context_past_kv = transfer_outputs.past_key_values

                elif cumulative_prompts[0]:
                    # Has text context but no latent hiddens (e.g. first agent generated text)
                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)

                    with torch.no_grad():
                        reenc_outputs = agent_model.model(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True
                        )
                        context_past_kv = reenc_outputs.past_key_values
                else:
                    # First agent, no prior context
                    context_past_kv = None

                _, new_latent_hiddens = self._capture_hidden_states_from_model(
                    agent_model, wrapped_ids, wrapped_mask, context_past_kv, self.latent_steps
                )

                if cumulative_latent_hiddens is None:
                    cumulative_latent_hiddens = new_latent_hiddens
                else:
                    cumulative_latent_hiddens = torch.cat(
                        [cumulative_latent_hiddens, new_latent_hiddens], dim=1
                    )
                current_model_name = agent_model_name

                for idx in range(batch_size):
                    cumulative_prompts[idx] += wrapped_prompts[idx]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    })
            else:
                # Last agent: rebuild context from hidden states, then generate text
                if cumulative_latent_hiddens is not None:
                    source_model = self.models[current_model_name]

                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)

                    with torch.no_grad():
                        prompt_embeds = agent_model.model.get_input_embeddings()(prompt_ids)

                    transferred_latent_embeds = transfer_via_realignment(
                        cumulative_latent_hiddens, source_model, agent_model
                    )

                    combined_embeds = torch.cat([prompt_embeds, transferred_latent_embeds], dim=1)
                    combined_mask = torch.cat([
                        prompt_mask,
                        torch.ones(
                            (batch_size, transferred_latent_embeds.shape[1]),
                            dtype=prompt_mask.dtype, device=prompt_mask.device)
                    ], dim=1)

                    with torch.no_grad():
                        transfer_outputs = agent_model.model(
                            inputs_embeds=combined_embeds,
                            attention_mask=combined_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True
                        )
                        past_for_decoding = transfer_outputs.past_key_values
                elif cumulative_prompts[0]:
                    prompt_encoded = agent_model.tokenizer(
                        cumulative_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    prompt_ids = prompt_encoded["input_ids"].to(agent_model.device)
                    prompt_mask = prompt_encoded["attention_mask"].to(agent_model.device)

                    with torch.no_grad():
                        reenc_outputs = agent_model.model(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            past_key_values=None,
                            use_cache=True,
                            return_dict=True
                        )
                        past_for_decoding = reenc_outputs.past_key_values
                else:
                    past_for_decoding = None

                if self.args.think:
                    final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    final_agent_prompts = prompts

                final_agent_encoded = agent_model.tokenizer(
                    final_agent_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                final_agent_ids = final_agent_encoded["input_ids"].to(agent_model.device)
                final_agent_mask = final_agent_encoded["attention_mask"].to(agent_model.device)
                final_agent_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(final_agent_ids, final_agent_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    final_agent_tokens_batch.append(agent_model.tokenizer.convert_ids_to_tokens(active_ids))

                generated_batch, _ = agent_model.generate_text_batch(
                    final_agent_ids,
                    final_agent_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = final_agent_mask[idx].bool()
                    trimmed_ids = final_agent_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": final_agent_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": final_agent_tokens_batch[idx],
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
