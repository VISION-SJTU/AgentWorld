"""Training-only influence ranking; all unspecified score scales are explicit defaults."""

import numpy as np


def select_agents(
    agents, ego_trajectory, limit=20, lane_half_width=2.0, safe_distance=4.0
):
    ego = np.asarray(ego_trajectory, dtype=float)
    scores = []
    for agent in agents:
        position = np.asarray(agent["box"][:2], dtype=float)
        velocity = np.asarray(agent.get("velocity", [0, 0]), dtype=float)
        distance = np.linalg.norm(position)
        current = (
            np.exp(-distance / 20.0)
            + np.exp(-abs(position[0]) / lane_half_width)
            + np.clip(-np.dot(position, velocity) / max(distance, 1.0) / 10.0, 0, 1)
        ) / 3
        mask = np.asarray(agent.get("mask", [True] * len(ego)), dtype=bool)
        relative = np.asarray(agent["trajectory"], dtype=float) - ego
        future_distance = np.linalg.norm(relative, axis=-1)
        if mask.any():
            minimum = future_distance[mask].min()
            unsafe_steps = np.flatnonzero(mask & (future_distance < safe_distance))
            ttc_score = 1 - (unsafe_steps[0] + 1) / len(ego) if len(unsafe_steps) else 0
            corridor = np.any(
                mask
                & (abs(relative[:, 0]) < lane_half_width)
                & (abs(relative[:, 1]) < safe_distance)
            )
            future = (
                np.exp(-minimum / safe_distance) + ttc_score + float(corridor)
            ) / 3
        else:
            future = 0.0
        category_weight = float(agent.get("influence_weight", 1.0))
        scores.append(category_weight * (0.45 * current + 0.55 * future))
    return [agents[i] for i in np.argsort(-np.asarray(scores), kind="stable")[:limit]]


def derive_action(
    trajectory,
    speed,
    dt=0.5,
    stop_speed=0.5,
    acceleration_threshold=0.5,
    lateral_threshold=2.0,
):
    points = np.asarray(trajectory, dtype=float)
    final_speed = np.linalg.norm(points[-1] - points[-2]) / dt
    acceleration = (final_speed - speed) / (len(points) * dt)
    longitudinal = (
        "STOP"
        if final_speed < stop_speed
        else "ACCELERATE"
        if acceleration > acceleration_threshold
        else "DECELERATE"
        if acceleration < -acceleration_threshold
        else "KEEP_SPEED"
    )
    lateral = (
        "CHANGE_LEFT"
        if points[-1, 0] < -lateral_threshold
        else "CHANGE_RIGHT"
        if points[-1, 0] > lateral_threshold
        else "KEEP_LANE"
    )
    return [longitudinal, lateral]
