"""Match Stage-1 slots to GT before creating token-referenced Stage-2 rationales."""

import argparse
import json

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams import DataArguments

from agentworld.backbone import GroundingContext
from agentworld.config import normalize_qwen_config
from agentworld.data import DrivingCollator, DrivingDataset
from agentworld.modules import AgentWorld, match_agents
from agentworld.tokens import gather_slots, perception_positions


def rationale(record, assignment):
    """Deterministic teacher; replace with offline MLLM annotations if available."""
    ego = np.asarray(record["ego_trajectory"])
    risks = []
    for slot, target in enumerate(assignment):
        if target is None:
            continue
        agent = record["agents"][target]
        mask = np.asarray(agent.get("mask", [True] * len(ego)), dtype=bool)
        if mask.any():
            clearance = np.linalg.norm(np.asarray(agent["trajectory"]) - ego, axis=-1)[mask].min()
            risks.append((clearance, slot + 1))
    relevant = sorted(risks)[:3]
    if not relevant:
        return "No immediate path conflict is predicted from active surrounding agents."
    mentions = ", ".join(f"<agent_{slot}>" for _, slot in relevant)
    if relevant[0][0] < 4:
        return f"{mentions} have limited predicted clearance to the ego path; maintain clearance and monitor their motion."
    return f"{mentions} are the nearest future participants, with no immediate path conflict predicted."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_max_pixels = 524288
    processor.image_min_pixels = 65536
    tokenizer = processor.tokenizer
    data_args = DataArguments(template="qwen2_vl", cutoff_len=8192)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    config = normalize_qwen_config(AutoConfig.from_pretrained(args.model))
    model = AutoModelForImageTextToText.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16)
    model.agentworld = AgentWorld.load(args.model).to(dtype=torch.bfloat16)
    model.to(args.device).eval()
    config = model.agentworld.config
    token_ids = tokenizer.convert_tokens_to_ids(config.tokens)
    encoder = SupervisedDatasetProcessor(template, tokenizer, processor, data_args)
    dataset = DrivingDataset(args.data, encoder, config, "grounding")
    collator = DrivingCollator(
        SFTDataCollatorWith4DAttentionMask(template=template, model=model, tokenizer=tokenizer, processor=processor)
    )
    context = GroundingContext(model, token_ids)
    try:
        with torch.no_grad(), open(args.output, "w") as output:
            for index, record in enumerate(dataset.records):
                batch = collator([dataset[index]])
                targets = [{k: v.to(args.device) for k, v in target.items()} for target in batch.pop("targets")]
                geometry, mask = batch.pop("geometry").to(args.device), batch.pop("geometry_mask").to(args.device)
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                assistant_ids = batch["input_ids"].masked_fill(batch["labels"].eq(-100), -1)
                positions, _ = perception_positions(
                    assistant_ids,
                    token_ids,
                    tokenizer.convert_tokens_to_ids("<perception>"),
                    tokenizer.convert_tokens_to_ids("</perception>"),
                )
                context.batch = (
                    geometry,
                    mask,
                    positions,
                    torch.ones(1, dtype=torch.bool, device=args.device),
                    batch["input_ids"],
                )
                try:
                    result = model(**batch, output_hidden_states=True, use_cache=False)
                finally:
                    context.clear()
                predicted = model.agentworld.head(gather_slots(result.hidden_states[-1], positions))
                rows, cols = match_agents({k: v[0] for k, v in predicted.items()}, targets[0])
                assignment = [None] * config.num_agents
                for row, col in zip(rows.tolist(), cols.tolist()):
                    assignment[row] = col
                record = dict(record)
                record["slot_assignment"] = assignment
                record["response"] = (
                    config.perception
                    + "<think>"
                    + rationale(record, assignment)
                    + "</think>"
                    + "<action>"
                    + ", ".join(record["actions"])
                    + "</action><answer>"
                    + json.dumps(record["ego_trajectory"])
                    + "</answer>"
                )
                record["images"] = [str((dataset.root / p).resolve()) for p in record["images"]]
                record["geometry"] = str((dataset.root / record["geometry"]).resolve())
                output.write(json.dumps(record) + "\n")
    finally:
        context.remove()


if __name__ == "__main__":
    main()
