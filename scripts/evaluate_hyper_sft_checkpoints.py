#!/usr/bin/env python3
"""Behavioral screen for HyPER-R1 SFT checkpoints.

This is deliberately not another validation-loss report. It asks each
checkpoint to generate the next graph action on held-out decision states and
measures syntax, action choice, arguments, depth behavior, and early commits.
Passing this screen only makes a checkpoint eligible for executable rollout
evaluation; it does not certify the checkpoint for RL training.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_PARSER_PATH = Path(__file__).parents[1] / "kbqa_r1" / "sexpr" / "action_parser.py"
_PARSER_SPEC = importlib.util.spec_from_file_location(
    "hyper_checkpoint_action_parser", _PARSER_PATH
)
if _PARSER_SPEC is None or _PARSER_SPEC.loader is None:
    raise ImportError(f"Cannot load action parser from {_PARSER_PATH}")
_PARSER_MODULE = importlib.util.module_from_spec(_PARSER_SPEC)
sys.modules[_PARSER_SPEC.name] = _PARSER_MODULE
_PARSER_SPEC.loader.exec_module(_PARSER_MODULE)
ActionParser = _PARSER_MODULE.ActionParser
ActionType = _PARSER_MODULE.ActionType


ActionSignature = Tuple[str, Tuple[str, ...]]


def action_signature(text: str) -> Optional[ActionSignature]:
    """Return one canonical tagged action, rejecting zero or multiple actions."""
    actions = ActionParser().parse_actions_from_text(text)
    if len(actions) != 1 or not actions[0].is_valid:
        return None
    action = actions[0]
    if action.source_span is None:
        return None
    for block in re.finditer(r"<action>(.*?)</action>", text, re.I | re.S):
        start, end = action.source_span
        overlap_start = max(block.start(1), start)
        overlap_end = min(block.end(1), end)
        parts = []
        if overlap_start >= overlap_end:
            parts.append(block.group(1))
        else:
            parts.extend(
                (
                    text[block.start(1) : overlap_start],
                    text[overlap_end : block.end(1)],
                )
            )
        if any(part.strip() for part in parts):
            return None
    arguments = tuple(argument.strip() for argument in action.arguments)
    if action.action_type == ActionType.COMBINE_HYPOTHESES:
        arguments = tuple(sorted(arguments))
    return action.action_type.value, arguments


def state_depth(messages: Sequence[Mapping[str, Any]]) -> int:
    """Read the deepest currently visible hypothesis from the public state."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        depths = [int(value) for value in re.findall(r"\bdepth=(\d+)\b", message.get("content", ""))]
        if depths:
            return max(depths)
    return -1


def depth_band(depth: int) -> str:
    if depth < 0:
        return "initial"
    if depth >= 2:
        return "depth_2_plus"
    return f"depth_{depth}"


def _stratum(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    target = action_signature(row["messages"][-1]["content"])
    return (
        str(row.get("extra_info", {}).get("family", "unknown")),
        depth_band(state_depth(row["messages"][:-1])),
        target[0] if target else "invalid_target",
    )


def stratified_sample(
    rows: Sequence[Mapping[str, Any]], limit: int, seed: int = 17
) -> List[Mapping[str, Any]]:
    """Round-robin across behavior strata so rare deep/recovery states survive."""
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratum(row)].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: List[Mapping[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit:
        advanced = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                advanced = True
        if not advanced:
            break
    return selected


def _weight_files_complete(model_dir: Path) -> Tuple[bool, str]:
    config = model_dir / "config.json"
    if not config.is_file():
        return False, "missing config.json"
    indices = list(model_dir.glob("*.index.json"))
    if indices:
        try:
            required = set()
            for index in indices:
                payload = json.loads(index.read_text(encoding="utf-8"))
                required.update(payload.get("weight_map", {}).values())
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"unreadable weight index: {exc}"
        missing = sorted(name for name in required if not (model_dir / name).is_file())
        if missing:
            return False, f"missing {len(missing)} indexed weight shard(s)"
        if required:
            return True, "complete indexed checkpoint"
    weights = [
        *model_dir.glob("*.safetensors"),
        *model_dir.glob("pytorch_model*.bin"),
    ]
    if not weights:
        return False, "missing model weights"
    if any(path.stat().st_size == 0 for path in weights):
        return False, "zero-byte model weight"
    return True, "complete checkpoint"


def resolve_checkpoint(path: Path) -> Path:
    candidates = [path, path / "huggingface", path / "hf_model"]
    for candidate in candidates:
        complete, _ = _weight_files_complete(candidate)
        if complete:
            return candidate
    reasons = [
        f"{candidate}: {_weight_files_complete(candidate)[1]}"
        for candidate in candidates
        if candidate.exists()
    ]
    raise ValueError(f"incomplete checkpoint {path}: " + "; ".join(reasons or ["path missing"]))


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "examples": len(rows),
        "parsable_single_action": _rate(rows, "parsable_single_action"),
        "action_type_accuracy": _rate(rows, "action_type_correct"),
        "exact_action_accuracy": _rate(rows, "exact_action"),
        "premature_commit_rate": _rate(rows, "premature_commit"),
        "predicted_actions": dict(sorted(Counter(row["predicted_type"] for row in rows).items())),
    }
    for field in ("family", "depth_band", "target_type", "decision_index"):
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        result[f"by_{field}"] = {
            key: {
                "examples": len(group),
                "parsable_single_action": _rate(group, "parsable_single_action"),
                "action_type_accuracy": _rate(group, "action_type_correct"),
                "exact_action_accuracy": _rate(group, "exact_action"),
                "premature_commit_rate": _rate(group, "premature_commit"),
            }
            for key, group in sorted(grouped.items())
        }
    deep = [row for row in rows if row["depth_band"] == "depth_2_plus"]
    gates = {
        "syntax_at_least_95pct": (result["parsable_single_action"] or 0.0) >= 0.95,
        "action_type_at_least_70pct": (result["action_type_accuracy"] or 0.0) >= 0.70,
        "exact_action_at_least_50pct": (result["exact_action_accuracy"] or 0.0) >= 0.50,
        "premature_commit_at_most_5pct": (result["premature_commit_rate"] or 0.0) <= 0.05,
        "deep_states_present": bool(deep),
        "deep_action_type_at_least_60pct": bool(deep) and (_rate(deep, "action_type_correct") or 0.0) >= 0.60,
        "deep_exact_action_at_least_40pct": bool(deep) and (_rate(deep, "exact_action") or 0.0) >= 0.40,
    }
    result["minimum_behavior_screen"] = {
        "passed": all(gates.values()),
        "checks": gates,
        "interpretation": (
            "Passing only permits executable rollout screening. It does not approve "
            "the checkpoint for RL or establish policy quality."
        ),
    }
    return result


def _load_rows(paths: Sequence[Path], per_data_limit: int) -> List[Dict[str, Any]]:
    import pyarrow.parquet as parquet

    selected: List[Dict[str, Any]] = []
    for data_path in paths:
        rows = parquet.read_table(data_path).to_pylist()
        for row in stratified_sample(rows, per_data_limit):
            copied = dict(row)
            copied["screen_dataset"] = data_path.stem
            selected.append(copied)
    return selected


def _prompts(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return [
        tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]


def evaluate_checkpoint(
    checkpoint: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer_path: Optional[Path],
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = resolve_checkpoint(checkpoint)
    tokenizer_source = tokenizer_path or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    predictions: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                _prompts(tokenizer, batch),
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            texts = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for row, prediction in zip(batch, texts):
                target_text = row["messages"][-1]["content"]
                target = action_signature(target_text)
                predicted = action_signature(prediction)
                target_type = target[0] if target else "invalid_target"
                predicted_type = predicted[0] if predicted else "unparsable"
                depth = state_depth(row["messages"][:-1])
                predictions.append(
                    {
                        "checkpoint": checkpoint.name,
                        "dataset": row["screen_dataset"],
                        "demo_id": row.get("extra_info", {}).get("demo_id"),
                        "family": row.get("extra_info", {}).get("family", "unknown"),
                        "decision_index": row.get("extra_info", {}).get("decision_index", -1),
                        "state_depth": depth,
                        "depth_band": depth_band(depth),
                        "target_type": target_type,
                        "predicted_type": predicted_type,
                        "target_action": target,
                        "predicted_action": predicted,
                        "parsable_single_action": predicted is not None,
                        "action_type_correct": target is not None and predicted is not None and target[0] == predicted[0],
                        "exact_action": target is not None and target == predicted,
                        "premature_commit": predicted_type == "Commit" and target_type != "Commit",
                        "target_text": target_text,
                        "prediction": prediction,
                    }
                )

    metrics = {
        "checkpoint": str(checkpoint),
        "resolved_model_dir": str(model_dir),
        **summarize(predictions),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, predictions


def _write_report(output: Path, metrics: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# HyPER-R1 SFT Behavioral Screen",
        "",
        "This screen evaluates generated graph actions, not templated validation loss. "
        "A pass only admits a checkpoint to executable rollout evaluation.",
        "",
        "| Checkpoint | N | Parsable | Type correct | Exact action | Early commit | Screen |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        lines.append(
            "| {name} | {n} | {parse:.3f} | {kind:.3f} | {exact:.3f} | {early:.3f} | {status} |".format(
                name=Path(item["checkpoint"]).name,
                n=item["examples"],
                parse=item["parsable_single_action"] or 0.0,
                kind=item["action_type_accuracy"] or 0.0,
                exact=item["exact_action_accuracy"] or 0.0,
                early=item["premature_commit_rate"] or 0.0,
                status="PASS" if item["minimum_behavior_screen"]["passed"] else "REJECT",
            )
        )
    lines.extend(
        [
            "",
            "The thresholds are minimum competence checks for an RL initializer, not "
            "claims of final quality. Rejected checkpoints must not be rescued by lower "
            "validation loss or sunk training cost.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--samples-per-data", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = _load_rows(args.data, args.samples_per_data)
    if not rows:
        raise SystemExit("No checkpoint-screen examples were loaded")
    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for checkpoint in args.checkpoints:
            metrics, predictions = evaluate_checkpoint(
                checkpoint,
                rows,
                tokenizer_path=args.tokenizer,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            all_metrics.append(metrics)
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, indent=2))
    (args.output / "metrics.json").write_text(
        json.dumps({"checkpoints": all_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output, all_metrics)


if __name__ == "__main__":
    main()
