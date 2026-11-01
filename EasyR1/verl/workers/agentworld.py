"""Reward-only decoder for current-policy agent hidden states; no new policy weights."""

import torch

from agentworld.modules import AgentWorld
from agentworld.tokens import gather_slots, perception_positions


class AgentProbe:
    def __init__(self, path, tokenizer, device, threshold=0.5):
        scaffold = AgentWorld.load(path)
        self.config = scaffold.config
        self.head = scaffold.head.to(device).eval().requires_grad_(False)
        self.token_ids = tokenizer.convert_tokens_to_ids(self.config.tokens)
        if any(
            tokenizer.encode(t, add_special_tokens=False) != [i] for t, i in zip(self.config.tokens, self.token_ids)
        ):
            raise ValueError("RL tokenizer must come from the AgentWorld SFT export.")
        self.opening_id = tokenizer.convert_tokens_to_ids("<perception>")
        self.closing_id = tokenizer.convert_tokens_to_ids("</perception>")
        self.threshold = threshold

    @torch.no_grad()
    def decode(self, model, batch, response_start=0):
        """A padded batch of one sample, with standard HF multimodal inputs."""
        outputs = model(**batch, output_hidden_states=True, use_cache=False)
        positions, valid = perception_positions(
            batch["input_ids"][:, response_start:], self.token_ids, self.opening_id, self.closing_id, strict=False
        )
        positions = positions + response_start
        prediction = self.head(gather_slots(outputs.hidden_states[-1], positions).float())
        probability = prediction["logits"].float().softmax(-1)
        active = (1 - probability[..., -1] >= self.threshold) & valid
        return prediction["trajectories"].float(), active
