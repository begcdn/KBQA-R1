#!/usr/bin/env python3
"""Create a HyPER-R1 copy of an existing KBQA-R1 parquet dataset."""

import argparse
from pathlib import Path
import re

from kbqa_r1.hyper_prompt import augment_dataset_row, dataset_candidate_entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from datasets import Dataset

    source = Dataset.from_parquet(args.input)
    rows = []
    candidate_mids = 0
    labeled_candidate_mids = 0
    for row in source:
        for entity in dataset_candidate_entities(row):
            if not entity:
                continue
            name, identity = str(entity[0]), str(entity[-1])
            if re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", identity):
                candidate_mids += 1
                labeled_candidate_mids += int(name != identity)
        rows.append(augment_dataset_row(row))
    label_rate = (
        labeled_candidate_mids / candidate_mids if candidate_mids else 1.0
    )
    if label_rate < 0.95:
        raise RuntimeError(
            "HyPER-R1 requires readable candidate entities in RL prompts; "
            f"only {labeled_candidate_mids}/{candidate_mids} MID candidates have labels. "
            "Rebuild the source dataset with prepare_rl_dataset.py --use_odbc."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    print(
        f"Wrote {len(rows)} HyPER-R1 examples to {output}; "
        f"candidate MID label coverage={label_rate:.3f}"
    )


if __name__ == "__main__":
    main()
