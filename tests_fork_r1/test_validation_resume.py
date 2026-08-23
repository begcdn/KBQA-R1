import json

import torch

from verl.trainer.ppo.ray_trainer_kbqa import (
    RayPPOTrainer,
    _load_completed_validation_ids,
    _normalize_generated_batch_dtypes,
)


def test_incremental_generation_dump_is_resumable(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 0

    for sample_id in ("question-1", "question-2"):
        trainer._dump_generations(
            inputs=[sample_id],
            outputs=["answer"],
            gts=[["m.1"]],
            scores=[1.0],
            reward_extra_infos_dict={"reward": [1.0]},
            dump_path=str(tmp_path),
            metadata=[{"id": sample_id}],
            filename="progress.jsonl",
            append=True,
        )

    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [row["metadata"]["id"] for row in rows] == ["question-1", "question-2"]
    assert _load_completed_validation_ids(tmp_path / "progress.jsonl") == {
        "question-1",
        "question-2",
    }


def test_resume_loader_ignores_interrupted_final_line(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text('{"metadata": {"id": "complete"}}\n{"metadata":')

    assert _load_completed_validation_ids(path) == {"complete"}


def test_validation_dtype_normalization_preserves_fractional_answer_f1():
    batch = {
        "responses": torch.tensor([[1, 2]], dtype=torch.int32),
        "hyper_r1_commit_answer_f1": torch.tensor([0.5], dtype=torch.float32),
    }

    _normalize_generated_batch_dtypes(batch)

    assert batch["responses"].dtype == torch.long
    assert batch["hyper_r1_commit_answer_f1"].dtype == torch.float32
    assert batch["hyper_r1_commit_answer_f1"].item() == 0.5
