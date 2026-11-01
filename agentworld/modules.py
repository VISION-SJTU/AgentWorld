from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from .config import AgentConfig


class RoutingLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.gates = nn.ModuleList([nn.Linear(dim, 1) for _ in range(2)])
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(2)]
        )
        self.fuse = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, geometry, vision, geometry_mask=None, vision_mask=None):
        responses = []
        for adapter, gate, attention, memory, mask in zip(
            self.adapters,
            self.gates,
            self.attention,
            (geometry, vision),
            (geometry_mask, vision_mask),
        ):
            adapted = adapter(query) * gate(query).sigmoid()
            responses.append(
                attention(
                    adapted, memory, memory, key_padding_mask=mask, need_weights=False
                )[0]
            )
        return self.norm(query + self.fuse(torch.cat([query, *responses], dim=-1)))


class AgentHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared = nn.Sequential(
            nn.Linear(config.hidden_size, config.routing_dim), nn.GELU()
        )
        self.category = nn.Linear(
            config.routing_dim, config.num_classes + 1
        )  # last = no-object
        self.box = nn.Linear(config.routing_dim, config.box_dim)
        self.trajectory = nn.Linear(config.routing_dim, config.horizon * 2)

    def forward(self, hidden):
        state = self.shared(hidden)
        return {
            "logits": self.category(state),
            "boxes": self.box(state),
            "trajectories": self.trajectory(state).unflatten(
                -1, (self.config.horizon, 2)
            ),
        }


class AgentWorld(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.routing_dim)
        self.geometry = nn.Linear(config.geometry_dim, config.routing_dim)
        self.vision = nn.Linear(config.hidden_size, config.routing_dim)
        self.routing = nn.ModuleList(
            [
                RoutingLayer(config.routing_dim, config.routing_heads)
                for _ in range(config.routing_layers)
            ]
        )
        self.output = nn.Linear(config.routing_dim, config.hidden_size)
        self.head = AgentHead(config)

    def anchor(
        self, embeddings, geometry, vision, geometry_mask=None, vision_mask=None
    ):
        query = self.query(embeddings)
        geometry, vision = self.geometry(geometry), self.vision(vision)
        for layer in self.routing:
            query = layer(query, geometry, vision, geometry_mask, vision_mask)
        return self.output(query)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "agentworld_config.json")
        torch.save(self.state_dict(), directory / "agentworld.pt")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        module = cls(AgentConfig.load(directory / "agentworld_config.json"))
        module.load_state_dict(
            torch.load(
                directory / "agentworld.pt", map_location="cpu", weights_only=True
            )
        )
        return module


def match_agents(prediction, target):
    """DETR assignment. Targets have labels[K], boxes[K,8], trajectories[K,6,2], mask[K,6]."""
    logits, boxes, tracks = (
        prediction["logits"],
        prediction["boxes"],
        prediction["trajectories"],
    )
    if target["labels"].numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=logits.device)
        return empty, empty
    mask = target["mask"].to(tracks)
    trajectory_cost = (
        (tracks[:, None] - target["trajectories"][None]).abs() * mask[None, :, :, None]
    ).sum((-1, -2)) / (2 * mask.sum(-1).clamp_min(1))[None]
    cost = (
        -logits.softmax(-1)[:, target["labels"]]
        + torch.cdist(boxes.float(), target["boxes"].float(), p=1) / boxes.size(-1)
        + trajectory_cost.float()
    )
    row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(row, device=logits.device),
        torch.as_tensor(col, device=logits.device),
    )


def grounding_loss(prediction, targets):
    zero = (
        prediction["logits"].sum() * 0
        + prediction["boxes"].sum() * 0
        + prediction["trajectories"].sum() * 0
    )
    losses = dict.fromkeys(("cls", "box", "traj", "noobj"), zero)
    assignments = []
    for batch, target in enumerate(targets):
        pred = {k: v[batch] for k, v in prediction.items()}
        row, col = match_agents(pred, target)
        assignments.append((row, col))
        unmatched = torch.ones(
            pred["logits"].size(0), dtype=torch.bool, device=row.device
        )
        unmatched[row] = False
        if row.numel():
            losses["cls"] = losses["cls"] + F.cross_entropy(
                pred["logits"][row].float(), target["labels"][col]
            )
            losses["box"] = losses["box"] + F.l1_loss(
                pred["boxes"][row].float(), target["boxes"][col].float()
            )
            mask = target["mask"][col].to(pred["trajectories"])
            error = (
                pred["trajectories"][row] - target["trajectories"][col]
            ).abs() * mask[..., None]
            losses["traj"] = losses["traj"] + error.sum() / (2 * mask.sum()).clamp_min(
                1
            )
        if unmatched.any():
            noobj = torch.full(
                (int(unmatched.sum()),),
                pred["logits"].size(-1) - 1,
                device=row.device,
                dtype=torch.long,
            )
            losses["noobj"] = losses["noobj"] + F.cross_entropy(
                pred["logits"][unmatched].float(), noobj
            )
    loss = (
        losses["cls"] + losses["box"] + losses["traj"] + 0.1 * losses["noobj"]
    ) / len(targets)
    return loss, assignments
