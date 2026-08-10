#!/usr/bin/env python3
"""Build HyPER-R1 SFT data from successful executable rollout parquet files."""

import argparse
from pathlib import Path

from kbqa_r1.hyper_sft import convert_sft_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="SFT parquet built from rollout dumps")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import Dataset

    source = Dataset.from_parquet(args.input)
    rows = []
    skipped = 0
    for row in source:
        try:
            converted = convert_sft_row(row)
        except ValueError:
            skipped += 1
            continue
        if not any(
            "Select [" in str(message.get("content", ""))
            for message in converted["messages"]
            if isinstance(message, dict)
        ):
            skipped += 1
            continue
        rows.append(converted)

    if not rows:
        raise RuntimeError(
            "No executable HyPER-R1 traces found. Collect rollouts with "
            "hyper_r1.enable=true before building SFT data."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    print(f"Wrote {len(rows)} HyPER-R1 traces to {output}; skipped {skipped}")


if __name__ == "__main__":
    main()
