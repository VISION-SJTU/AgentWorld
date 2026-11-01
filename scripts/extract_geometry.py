"""Offline VGGT patch memory from current/past views only.

Install facebookresearch/vggt separately. This intentionally calls the official
aggregator, not the prediction heads; no future labels or images are consumed.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from agentworld.data import read_records


def main():
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--model", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.data).resolve().parent
    model = (
        VGGT.from_pretrained(args.model).to(args.device).eval().requires_grad_(False)
    )
    with torch.no_grad():
        for record in read_records(args.data):
            path = root / record["geometry"]
            if path.exists():
                continue
            image_paths = [
                str((root / p).resolve())
                for p in record.get("geometry_images", record["images"])
            ]
            images = load_and_preprocess_images(image_paths).to(args.device)[None]
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.device.startswith("cuda")
                else nullcontext()
            )
            with autocast:
                layers, patch_start = model.aggregator(images)
            features = layers[-1][:, :, patch_start:]
            features = features.reshape(-1, features.size(-1)).half().cpu()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, path)


if __name__ == "__main__":
    main()
