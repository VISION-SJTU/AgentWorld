"""Normalize prepared ego-frame records and select influential agents (training only)."""

import argparse
import json
from pathlib import Path

from agentworld.data import read_records
from agentworld.selection import derive_action, select_agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--num-agents", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.input).resolve().parent
    with open(args.output, "w") as output:
        for record in read_records(args.input):
            record["images"] = [
                str((root / path).resolve()) for path in record["images"]
            ]
            record["geometry"] = str((root / record["geometry"]).resolve())
            record["agents"] = select_agents(
                record["agents"], record["ego_trajectory"], args.num_agents
            )
            record.setdefault(
                "actions", derive_action(record["ego_trajectory"], record["speed"])
            )
            output.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
