import torch


def register_tokens(tokenizer, config):
    # Agent IDs are special tokens, but must NOT become generation EOS IDs.
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": config.tokens
            + [
                "<perception>",
                "</perception>",
                "<action>",
                "</action>",
                "<answer>",
                "</answer>",
            ]
        },
        replace_additional_special_tokens=False,
    )
    ids = tokenizer.convert_tokens_to_ids(config.tokens)
    if len(set(ids)) != config.num_agents or any(
        tokenizer.encode(t, add_special_tokens=False) != [i]
        for t, i in zip(config.tokens, ids)
    ):
        raise ValueError("Each agent must have its own atomic token ID.")
    return ids


def gather_slots(hidden, positions):
    return hidden.gather(1, positions[..., None].expand(-1, -1, hidden.size(-1)))


def perception_positions(input_ids, token_ids, opening_id, closing_id, strict=True):
    """Inspect one perception block in the supplied assistant region; mask malformed rollouts."""
    positions = torch.zeros(
        (input_ids.size(0), len(token_ids)), dtype=torch.long, device=input_ids.device
    )
    valid = torch.zeros_like(positions, dtype=torch.bool)
    for b, sequence in enumerate(input_ids):
        opens = sequence.eq(opening_id).nonzero().flatten()
        closes = sequence.eq(closing_id).nonzero().flatten()
        if opens.numel() != 1 or closes.numel() != 1 or closes[0] <= opens[0]:
            if strict:
                raise ValueError("Expected exactly one complete perception block.")
            continue
        start, end = int(opens[0]) + 1, int(closes[0])
        region = sequence[start:end]
        for i, token in enumerate(token_ids):
            found = region.eq(token).nonzero().flatten()
            if found.numel() == 1:
                positions[b, i] = found[0] + start
                valid[b, i] = True
        if strict and (not valid[b].all() or end - start != len(token_ids)):
            raise ValueError(
                "Perception must contain each agent token exactly once, with no extra text."
            )
    return positions, valid
