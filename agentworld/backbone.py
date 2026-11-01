"""Qwen2/2.5-VL hooks: inject slots after visual fusion, before the text decoder.

No attention or decoder implementation is replaced. Hooks are active only during
an explicit grounding forward and are removed from deployment exports.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from .tokens import gather_slots


class GroundingContext:
    def __init__(self, model, token_ids):
        self.model = model
        self.token_ids = token_ids
        self.batch = None
        self.vision = None
        base = model.model
        if hasattr(base, "language_model"):  # transformers >=4.52 Qwen VL layout
            visual, decoder = base.visual, base.language_model
        elif hasattr(model, "visual"):  # older Qwen2-VL layout
            visual, decoder = model.visual, base
        else:
            raise ValueError(
                "Grounding currently supports Qwen2-VL and Qwen2.5-VL only."
            )
        self.handles = [
            visual.register_forward_hook(self._capture_vision),
            decoder.register_forward_pre_hook(self._inject, with_kwargs=True),
        ]

    def _capture_vision(self, module, args, output):
        if self.batch is None:
            return
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        if isinstance(features, tuple):
            features = features[0]
        self.vision = features

    def _inject(self, module, args, kwargs):
        if self.batch is None:
            return
        geometry, geometry_mask, positions, keep, input_ids = self.batch
        if self.vision is None:
            raise ValueError("Grounding needs real images and visual encoder features.")
        config = self.model.agentworld.config
        counts = input_ids.eq(config.image_token_id).sum(-1).tolist()
        if sum(counts) != self.vision.size(0) or not all(counts):
            raise ValueError(
                "Visual feature counts do not match per-sample image tokens."
            )
        chunks = self.vision.split(counts)
        vision = pad_sequence(chunks, batch_first=True)
        lengths = torch.as_tensor(counts, device=vision.device)
        vision_mask = (
            torch.arange(vision.size(1), device=vision.device)[None] >= lengths[:, None]
        )
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            raise ValueError(
                "Expected fused inputs_embeds at the text decoder boundary."
            )
        # Use the special token embedding itself as query, not an independent query bank.
        agent_ids = torch.as_tensor(self.token_ids, device=embeddings.device)
        query = self.model.get_input_embeddings()(agent_ids)[None].expand(
            input_ids.size(0), -1, -1
        )
        anchored = self.model.agentworld.anchor(
            query, geometry.to(query), vision.to(query), geometry_mask, vision_mask
        )
        original = gather_slots(embeddings, positions)
        replacement = torch.where(keep[:, None, None], anchored, original)
        kwargs["inputs_embeds"] = embeddings.scatter(
            1, positions[..., None].expand_as(replacement), replacement.to(embeddings)
        )
        return args, kwargs

    def clear(self):
        self.batch = None
        self.vision = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.clear()
