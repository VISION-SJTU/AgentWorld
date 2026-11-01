"""Prepared JSONL records; physical targets never enter policy text."""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_records(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def target_tensors(record, config):
    objects = record["agents"]
    count = len(objects)
    if count > config.num_agents:
        raise ValueError("Run influential-agent selection before training.")
    labels = torch.tensor([o["label"] for o in objects], dtype=torch.long)
    boxes = torch.tensor([o["box"] for o in objects], dtype=torch.float32).reshape(
        count, config.box_dim
    )
    tracks = torch.tensor(
        [o["trajectory"] for o in objects], dtype=torch.float32
    ).reshape(count, config.horizon, 2)
    mask = torch.tensor(
        [o.get("mask", [True] * config.horizon) for o in objects], dtype=torch.bool
    ).reshape(count, config.horizon)
    if labels.numel() and ((labels < 0).any() or (labels >= config.num_classes).any()):
        raise ValueError("Invalid class label.")
    if not torch.isfinite(boxes).all() or not torch.isfinite(tracks).all():
        raise ValueError(
            "Replace missing trajectory coordinates with zero and mask them out."
        )
    return {"labels": labels, "boxes": boxes, "trajectories": tracks, "mask": mask}


class DrivingDataset(Dataset):
    def __init__(self, path, encoder, config, stage):
        self.records = read_records(path)
        self.root = Path(path).resolve().parent
        self.encoder, self.config, self.stage = encoder, config, stage

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = [str((self.root / path).resolve()) for path in record["images"]]
        prompt = record["prompt"]
        if prompt.count("<image>") != len(images):
            raise ValueError(
                "One <image> placeholder is required per image, in camera order."
            )
        response = (
            self.config.perception if self.stage == "grounding" else record["response"]
        )
        if not response.startswith(self.config.perception):
            raise ValueError(
                "Assistant response must begin with the fixed perception block."
            )
        ids, labels = self.encoder._encode_data_example(
            prompt=[{"role": "user", "content": prompt}],
            response=[{"role": "assistant", "content": response}],
            system=record.get("system", ""),
            tools="",
            images=images,
            videos=[],
            audios=[],
        )
        geometry = torch.load(
            self.root / record["geometry"], map_location="cpu", weights_only=True
        )
        if (
            geometry.ndim != 2
            or geometry.size(1) != self.config.geometry_dim
            or not geometry.size(0)
        ):
            raise ValueError(
                "Geometry cache must be a nonempty [memory_tokens, geometry_dim] tensor."
            )
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
            "images": images,
            "videos": [],
            "audios": [],
            "geometry": geometry.float(),
            "targets": target_tensors(record, self.config),
        }


class DrivingCollator:
    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        features = [dict(feature) for feature in features]
        geometry = [f.pop("geometry") for f in features]
        targets = [f.pop("targets") for f in features]
        batch = self.base(features)
        batch["geometry"] = pad_sequence(geometry, batch_first=True)
        lengths = torch.tensor([g.size(0) for g in geometry])
        batch["geometry_mask"] = (
            torch.arange(batch["geometry"].size(1))[None] >= lengths[:, None]
        )
        batch["targets"] = targets
        return batch
