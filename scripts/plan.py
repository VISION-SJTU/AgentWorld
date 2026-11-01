"""Standard MLLM-only inference from one record's visible images/prompt."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from agentworld.data import read_records
from agentworld.rewards import RewardConfig, parse_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-agents", type=int, default=20)
    args = parser.parse_args()
    record = read_records(args.data)[args.index]
    root = Path(args.data).resolve().parent
    processor = AutoProcessor.from_pretrained(args.model)
    images = [str((root / p).resolve()) for p in record["images"]]
    parts = record["prompt"].split("<image>")
    if len(parts) - 1 != len(images):
        raise ValueError("Image placeholders must match image paths.")
    content = []
    for i, text in enumerate(parts):
        if i:
            content.append({"type": "image", "url": images[i - 1]})
        if text:
            content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    if record.get("system"):
        messages.insert(0, {"role": "system", "content": record["system"]})
    # Uses only visible inputs: no geometry cache, target states, head or routing.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(args.device)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = (
        AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype)
        .to(args.device)
        .eval()
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    response = processor.tokenizer.decode(
        generated[0, inputs["input_ids"].size(1) :], skip_special_tokens=False
    )
    parsed = parse_response(response, RewardConfig(num_agents=args.num_agents))
    if parsed is None or not parsed["format"]:
        raise ValueError("Invalid structured policy response: " + response)
    print(
        json.dumps(
            {
                "response": response,
                "actions": sorted(parsed["actions"]),
                "trajectory": parsed["trajectory"].tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
