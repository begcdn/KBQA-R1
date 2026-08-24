#!/usr/bin/env python3
"""Fail before SFT if the exact tokenizer would clip a HyPER target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kbqa_r1.hyper_r1 import render_hyper_observation_suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chat-template")
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    if args.chat_template:
        tokenizer.chat_template = Path(args.chat_template).read_text(
            encoding="utf-8"
        )
    rows = 0
    maximum = 0
    for path in args.data:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["messages"], batch_size=256):
            conversations = []
            for offset, record in enumerate(batch.to_pylist()):
                messages = record["messages"]
                supervised = [
                    message
                    for message in messages
                    if int(message.get("loss_mask") or 0) == 1
                ]
                if (
                    len(supervised) != 1
                    or supervised[0].get("role") != "assistant"
                    or supervised[0] is not messages[-1]
                    or not str(supervised[0].get("content", "")).strip()
                ):
                    raise RuntimeError(
                        f"row {rows + offset} does not end in exactly one supervised action"
                    )
                context = [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages[:-1]
                ]
                sft_context = tokenizer.apply_chat_template(
                    context,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                initial = tokenizer.apply_chat_template(
                    context[:1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if len(context) == 1:
                    runtime_context = initial
                elif (
                    len(context) == 3
                    and context[1] == {"role": "assistant", "content": ""}
                    and context[2]["role"] == "user"
                ):
                    runtime_context = initial + render_hyper_observation_suffix(
                        tokenizer, context[2]["content"]
                    )
                else:
                    raise RuntimeError(
                        f"row {rows + offset} has a non-Markov prompt shape"
                    )
                if runtime_context != sft_context:
                    raise RuntimeError(
                        f"row {rows + offset} differs from the live HyPER prompt"
                    )
                conversations.append(messages)
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
            )
            sequences = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
            for offset, input_ids in enumerate(sequences):
                length = len(input_ids)
                if length > args.max_length:
                    raise RuntimeError(
                        f"row {rows + offset} has {length} tokens, above {args.max_length}"
                    )
                maximum = max(maximum, length)
            rows += len(conversations)
    print(
        json.dumps(
            {
                "rows": rows,
                "maximum_tokens": maximum,
                "limit": args.max_length,
                "overlength": 0,
                "zero_target": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
