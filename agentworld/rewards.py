"""Text rewards and Eq.21-24. No ground-truth agents are used for interaction reward."""

import ast
from dataclasses import dataclass
import json
import re

import numpy as np


LONGITUDINAL = {"ACCELERATE", "KEEP_SPEED", "DECELERATE", "STOP"}
LATERAL = {"KEEP_LANE", "CHANGE_LEFT", "CHANGE_RIGHT"}


@dataclass
class RewardConfig:
    num_agents: int = 20
    horizon: int = 6
    dt: float = 0.5
    d_col: float = 2.0
    d_safe: float = 4.0
    t_safe: float = 3.0
    lambda_col: float = 1.0
    lambda_clear: float = 1.0
    lambda_ttc: float = 1.0
    trajectory_scale: float = 1.0
    smoothness_scale: float = 1.0
    existence_threshold: float = 0.5
    top_k: int = 20


def parse_response(response, config):
    # Preserve agent special tokens; remove only chat-level generation terminators.
    response = (
        response.strip()
        .removesuffix("<|im_end|>")
        .strip()
        .removesuffix("<|endoftext|>")
        .strip()
    )
    pattern = r"<perception>(.*?)</perception>\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*<answer>(.*?)</answer>"
    match = re.fullmatch(pattern, response, flags=re.DOTALL)
    if not match:
        return None
    perception, think, action, answer = match.groups()
    tokens = re.findall(r"<agent_\d+>", perception)
    expected = {f"<agent_{i + 1}>" for i in range(config.num_agents)}
    valid_perception = (
        len(tokens) == config.num_agents
        and set(tokens) == expected
        and "".join(tokens) == perception.strip()
    )
    actions = [part.strip() for part in action.split(",")]
    valid_action = (
        len(actions) == 2
        and len(set(actions) & LONGITUDINAL) == 1
        and len(set(actions) & LATERAL) == 1
    )
    try:
        trajectory = np.asarray(ast.literal_eval(answer.strip()), dtype=float)
        if trajectory.shape != (config.horizon, 2) or not np.isfinite(trajectory).all():
            return None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return {
        "perception": float(valid_perception),
        "format": float(valid_perception and valid_action),
        "actions": set(actions) if valid_action else set(),
        "trajectory": trajectory,
        "references": float(any(t in think for t in expected)),
    }


def interaction_reward(ego, tracks, valid, config):
    tracks, valid = np.asarray(tracks, dtype=float), np.asarray(valid, dtype=bool)
    if tracks.shape != (config.num_agents, config.horizon, 2) or valid.shape != (
        config.num_agents,
    ):
        raise ValueError("Expected decoded [N,H,2] tracks and [N] active-slot mask.")
    selected = tracks[valid]
    if not len(selected):
        return 0.0
    if not np.isfinite(selected).all():
        raise ValueError("Non-finite decoded agent trajectory.")
    distance = np.linalg.norm(selected - ego[None], axis=-1)
    distance = distance[np.argsort(distance.min(-1), kind="stable")[: config.top_k]]
    minimum = distance.min(-1)
    unsafe = distance < config.d_safe
    ttc = np.where(unsafe.any(-1), (unsafe.argmax(-1) + 1) * config.dt, np.inf)
    penalty = (
        config.lambda_col * (minimum < config.d_col)
        + config.lambda_clear * np.maximum(1 - minimum / config.d_safe, 0)
        + config.lambda_ttc * np.maximum(1 - ttc / config.t_safe, 0)
    )
    return -float(penalty.mean())


def compute_score(item, **kwargs):
    config = RewardConfig(**kwargs)
    truth = (
        json.loads(item["ground_truth"])
        if isinstance(item["ground_truth"], str)
        else item["ground_truth"]
    )
    parsed = parse_response(item["response"], config)
    metrics = {
        key: 0.0
        for key in (
            "format",
            "perception",
            "action",
            "trajectory",
            "agent",
            "references",
        )
    }
    if parsed is None:
        return {"overall": -1.0, **metrics}  # explicit invalid-output convention
    expert = np.asarray(truth["trajectory"], dtype=float)
    if expert.shape != (config.horizon, 2) or not np.isfinite(expert).all():
        raise ValueError("Invalid expert ego trajectory.")
    predicted, reference = parsed["actions"], set(truth["actions"])
    metrics.update({key: parsed[key] for key in ("format", "perception", "references")})
    metrics["action"] = (
        2 * len(predicted & reference) / max(len(predicted) + len(reference), 1)
    )
    # Paper does not specify R_traj. Explicit engineering choice: bounded ADE + acceleration error.
    ade = np.linalg.norm(parsed["trajectory"] - expert, axis=-1).mean()
    origin = np.zeros((1, 2))
    acceleration = (
        np.diff(np.concatenate([origin, parsed["trajectory"]]), n=2, axis=0)
        / config.dt**2
    )
    reference_acceleration = (
        np.diff(np.concatenate([origin, expert]), n=2, axis=0) / config.dt**2
    )
    smoothness_error = np.linalg.norm(
        acceleration - reference_acceleration, axis=-1
    ).mean()
    metrics["trajectory"] = float(
        np.exp(
            -ade / config.trajectory_scale - smoothness_error / config.smoothness_scale
        )
    )
    # The actor worker must supply current-policy hidden-state predictions; never substitute GT tracks.
    metrics["agent"] = interaction_reward(
        parsed["trajectory"], item["agent_trajectories"], item["agent_valid"], config
    )
    overall = sum(
        metrics[key]
        for key in ("format", "perception", "action", "trajectory", "agent")
    )
    return {"overall": float(overall), **metrics}
