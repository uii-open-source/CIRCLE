from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3TextEncoder(nn.Module):
    """
    Light-weight Qwen3 text feature extractor.
    Uses HF transformers `AutoModel` / `AutoTokenizer`.

    Shape:
        input: list[str] (length = batch_size)
        output: (batch_size, hidden_size)          — global pooled embedding
                and (batch_size, seq_len, hidden_size) — per-token embedding
    """

    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-0.5B",
                 freeze: bool = True,
                 max_seq_len: int = 256,
                 use_hf_token: bool = False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.max_seq_len = max_seq_len
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._frozen = freeze

    @torch.no_grad()
    def encode_tokens(self, prompts: List[str], device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize prompts and return input_ids + attention mask.

        Args:
            prompts: list of text strings
            device: torch device
        Returns:
            input_ids: (B, L) tensor
            attention_mask: (B, L) tensor
        """
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        return input_ids, attention_mask

    def forward(self, prompts: List[str], device=None):
        """
        Forward pass: tokenize -> transformer -> mean-pool -> projector.

        Args:
            prompts: list of text strings (batch = len(prompts))
            device: optional torch device
        Returns:
            projected: (B, D) global pooled and projected text embeddings
            last_hidden: (B, L, D) per-token last-layer hidden states
            attention_mask: (B, L) token mask (1 = valid, 0 = pad)
        """
        if device is None:
            device = next(self.parameters()).device

        input_ids, attention_mask = self.encode_tokens(prompts, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state.float()

        mask = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * mask).sum(dim=1)
        count_emb = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_emb / count_emb

        projected = mean_pooled

        return projected, last_hidden, attention_mask
