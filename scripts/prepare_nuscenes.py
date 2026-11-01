"""nuScenes train/val annotations -> AgentWorld's current-ego-frame JSONL schema.

Requires the official nuscenes-devkit. Future GT is written only to supervision;
geometry images contain current/past observations, never future frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from agentworld.selection import derive_action, select_agents


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def ego_pose(nusc, sample):
    data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", data["ego_pose_token"])


def interpolate_track(times, positions, desired_times):
    times, positions = np.asarray(times), np.asarray(positions)
    valid = (desired_times >= times[0]) & (desired_times <= times[-1])
    points = np.stack(
        [np.interp(desired_times, times, positions[:, axis]) for axis in range(3)],
        axis=-1,
    )
    return points, valid


def main():
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--split", choices=["train", "val", "mini_train", "mini_val"], required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument(
        "--history",
        type=int,
        default=1,
        help="Number of past keyframes for geometry, not future frames.",
    )
    parser.add_argument("--with-ego-state", action="store_true")
    parser.add_argument(
        "--navigation", help="Optional JSON mapping sample_token -> route command."
    )
    args = parser.parse_args()
    navigation = (
        json.loads(Path(args.navigation).read_text()) if args.navigation else {}
    )
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = set(create_splits_scenes()[args.split])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_root, geometry_root = (
        Path(args.dataroot).resolve(),
        Path(args.geometry_dir).resolve(),
    )
    # Standard nuScenes ego axes: x forward, y left. New axes: x right, y forward.
    axes = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with output_path.open("w") as output:
        for sample in nusc.sample:
            if nusc.get("scene", sample["scene_token"])["name"] not in scenes:
                continue
            pose = ego_pose(nusc, sample)
            rotation = axes @ Quaternion(pose["rotation"]).rotation_matrix.T
            origin = np.asarray(pose["translation"])

            def transform(points):
                return (np.asarray(points) - origin) @ rotation.T

            timestamp = sample["timestamp"] / 1e6
            desired = timestamp + np.arange(1, 7) * 0.5
            ego_times, ego_points = [timestamp], [pose["translation"]]
            cursor = sample
            while cursor["next"] and ego_times[-1] < desired[-1]:
                cursor = nusc.get("sample", cursor["next"])
                ego_times.append(cursor["timestamp"] / 1e6)
                ego_points.append(ego_pose(nusc, cursor)["translation"])
            future_ego, ego_valid = interpolate_track(ego_times, ego_points, desired)
            if not ego_valid.all():
                continue
            ego_trajectory = transform(future_ego)[:, :2].tolist()
            speed = 0.0
            if sample["prev"]:
                previous = nusc.get("sample", sample["prev"])
                delta = timestamp - previous["timestamp"] / 1e6
                speed = float(
                    np.linalg.norm(
                        np.asarray(pose["translation"])[:2]
                        - np.asarray(ego_pose(nusc, previous)["translation"])[:2]
                    )
                    / max(delta, 1e-6)
                )
            agents = []
            for annotation_id in sample["anns"]:
                annotation = nusc.get("sample_annotation", annotation_id)
                category = category_to_detection_name(annotation["category_name"])
                if category not in CLASSES:
                    continue
                position = transform(annotation["translation"])
                heading = (
                    rotation @ Quaternion(annotation["rotation"]).rotation_matrix[:, 0]
                )
                yaw = np.arctan2(heading[1], heading[0])
                velocity = rotation @ np.nan_to_num(
                    nusc.box_velocity(annotation_id), nan=0.0
                )
                times, points = [timestamp], [annotation["translation"]]
                cursor = annotation
                while cursor["next"] and times[-1] < desired[-1]:
                    cursor = nusc.get("sample_annotation", cursor["next"])
                    times.append(
                        nusc.get("sample", cursor["sample_token"])["timestamp"] / 1e6
                    )
                    points.append(cursor["translation"])
                future, valid = interpolate_track(times, points, desired)
                trajectory = transform(future)[:, :2]
                trajectory[~valid] = 0
                agents.append(
                    {
                        "instance_token": annotation["instance_token"],
                        "label": CLASSES.index(category),
                        "box": [
                            *position.tolist(),
                            *annotation["size"],
                            float(np.sin(yaw)),
                            float(np.cos(yaw)),
                        ],
                        "velocity": velocity[:2].tolist(),
                        "trajectory": trajectory.tolist(),
                        "mask": valid.tolist(),
                        "influence_weight": 1.5
                        if category in ("pedestrian", "bicycle", "motorcycle")
                        else 1.0,
                    }
                )
            images = [
                str(
                    data_root
                    / nusc.get("sample_data", sample["data"][camera])["filename"]
                )
                for camera in CAMERAS
            ]
            history = []
            cursor = sample
            for _ in range(args.history):
                if not cursor["prev"]:
                    break
                cursor = nusc.get("sample", cursor["prev"])
                history.append(cursor)
            geometry_images = [
                str(
                    data_root
                    / nusc.get("sample_data", frame["data"][camera])["filename"]
                )
                for frame in [*reversed(history), sample]
                for camera in CAMERAS
            ]
            prompt = (
                "<image>"
                * len(images)
                + "\nYou are the decision-making AI for an autonomous vehicle. "
                "Coordinates: X is right, Y is forward, ego is (0,0), units are meters. "
                "Plan 6 waypoints at 0.5-second intervals for the next 3 seconds. "
                "Use <perception> agent slots, then <think>, <action>, and <answer> fields."
            )
            if sample["token"] in navigation:
                prompt += " Navigation: " + str(navigation[sample["token"]]) + "."
            if args.with_ego_state:
                prompt += f" Current speed: {speed:.2f} m/s."
            record = {
                "id": sample["token"],
                "images": images,
                "geometry_images": geometry_images,
                "geometry": str(geometry_root / (sample["token"] + ".pt")),
                "prompt": prompt,
                "speed": speed,
                "ego_trajectory": ego_trajectory,
                "actions": derive_action(ego_trajectory, speed),
                "agents": select_agents(agents, ego_trajectory),
                "ego_state_in_prompt": args.with_ego_state,
            }
            output.write(json.dumps(record) + "\n")
    output_path.with_suffix(".classes.json").write_text(json.dumps(CLASSES, indent=2))


if __name__ == "__main__":
    main()
