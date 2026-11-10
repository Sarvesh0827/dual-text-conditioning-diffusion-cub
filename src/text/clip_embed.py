from __future__ import annotations

import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from src.text.condition import TextCondition


class CLIPTextEmbedder(nn.Module):
    """Frozen CLIP ViT-B/32 text encoder. Raw last-hidden and EOS pooled features."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, captions: list[str], device: torch.device | None = None) -> TextCondition:
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.model(**tokens)
        # CLIP EOS pooled representation used by CLIP contrastive head.
        input_ids = tokens["input_ids"]
        eos_mask = input_ids == self.tokenizer.eos_token_id
        # last EOS per sequence
        eos_indices = eos_mask.long().argmax(dim=1)
        b = input_ids.size(0)
        pooled = out.last_hidden_state[torch.arange(b, device=out.last_hidden_state.device), eos_indices]
        return TextCondition(
            tokens=out.last_hidden_state,
            pooled=pooled,
            input_ids=input_ids,
            attention_mask=tokens.get("attention_mask"),
        )
