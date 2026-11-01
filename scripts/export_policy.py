"""Create a standard HF/vLLM policy directory plus a separate frozen reward scaffold."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("policy")
    parser.add_argument("reward_scaffold")
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    destination = Path(args.policy).resolve()
    reward = Path(args.reward_scaffold).resolve()
    if (
        source in (destination, reward)
        or destination == reward
        or destination.exists()
        or reward.exists()
    ):
        raise ValueError(
            "Choose distinct, new output directories; existing checkpoints are never overwritten."
        )
    for filename in ("config.json", "agentworld.pt", "agentworld_config.json"):
        if not (source / filename).exists():
            raise FileNotFoundError(source / filename)
    destination.mkdir(parents=True)
    reward.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in ("agentworld.pt", "agentworld_config.json"):
            shutil.copy2(path, reward / path.name)
        elif path.is_file() and (
            path.suffix
            in (".json", ".safetensors", ".model", ".jinja", ".txt", ".tiktoken")
            or path.name.startswith("pytorch_model")
        ):
            if path.name not in (
                "trainer_state.json",
                "train_results.json",
                "all_results.json",
            ):
                shutil.copy2(path, destination / path.name)

    # Older Qwen-VL configs proxy all nested instance attributes, including model_type.
    # Omit the redundant nested type so standard HF/vLLM loaders see the outer VL type.
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") in ("qwen2_vl", "qwen2_5_vl"):
        config.get("text_config", {}).pop("model_type", None)
        config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
