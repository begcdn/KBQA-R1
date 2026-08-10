#!/usr/bin/env python3
"""Create a HyPER-R1 copy of an existing KBQA-R1 parquet dataset."""

import argparse
from pathlib import Path

from kbqa_r1.hyper_prompt import augment_dataset_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import Dataset

    source = Dataset.from_parquet(args.input)
    rows = [augment_dataset_row(row) for row in source]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    print(f"Wrote {len(rows)} HyPER-R1 examples to {output}")


if __name__ == "__main__":
    main()

