"""Convert Stage-2 records to EasyR1 JSONL, keeping labels out of prompts."""

import argparse
import json
from pathlib import Path

from agentworld.data import read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.input).resolve().parent
    with open(args.output, "w") as output:
        for record in read_records(args.input):
            output.write(
                json.dumps(
                    {
                        "prompt": record["prompt"],
                        "system": record.get("system", ""),
                        "images": [str((root / p).resolve()) for p in record["images"]],
                        "answer": json.dumps(
                            {
                                "actions": record["actions"],
                                "trajectory": record["ego_trajectory"],
                            }
                        ),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
