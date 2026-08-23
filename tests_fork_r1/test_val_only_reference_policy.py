from omegaconf import OmegaConf
import numpy as np
import torch

from verl import DataProto
from verl.trainer.main_ppo_kbqa import TaskRunner
from verl.trainer.ppo.ray_trainer_kbqa import RayPPOTrainer, Role
from verl.workers.fsdp_workers import direct_validation_load_format


def test_val_only_does_not_create_reference_policy_worker():
    task_runner_class = TaskRunner.__ray_metadata__.modified_class
    runner = task_runner_class.__new__(task_runner_class)
    runner.role_worker_mapping = {}
    runner.mapping = {}
    config = OmegaConf.create(
        {
            "trainer": {"val_only": True},
            "algorithm": {"use_kl_in_reward": True},
            "actor_rollout_ref": {"actor": {"use_kl_loss": True}},
        }
    )

    runner.add_ref_policy_worker(config, object)

    assert runner.role_worker_mapping == {}
    assert runner.mapping == {}


def test_val_only_initializes_actor_rollout_mapping_as_rollout(monkeypatch):
    import verl.trainer.ppo.ray_trainer_kbqa as trainer_module

    captured = {}

    class FakeInitArgs:
        def __init__(self, cls, config, role):
            captured["role"] = role
            captured["validation_only"] = config.get("validation_only", False)

    class FakeResourcePoolManager:
        resource_pool_dict = {"global": "global"}

        def create_resource_pool(self):
            pass

        def get_resource_pool(self, role):
            assert role == Role.ActorRollout
            return "global"

    monkeypatch.setattr(trainer_module, "RayClassWithInitArgs", FakeInitArgs)

    trainer = trainer_module.RayPPOTrainer.__new__(trainer_module.RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"val_only": True},
            "actor_rollout_ref": {},
            "global_profiler": {"steps": None},
        }
    )
    trainer.resource_pool_manager = FakeResourcePoolManager()
    trainer.role_worker_mapping = {Role.ActorRollout: object()}
    trainer.hybrid_engine = True
    trainer.use_critic = False
    trainer.use_reference_policy = False
    trainer.use_rm = False
    trainer.device_name = "cuda"

    class StopAfterRoleCapture(Exception):
        pass

    def stop_before_spawning(*args, **kwargs):
        raise StopAfterRoleCapture

    monkeypatch.setattr(trainer_module, "create_colocated_worker_cls", stop_before_spawning)

    try:
        trainer.init_workers()
    except StopAfterRoleCapture:
        pass

    assert captured["role"] == "rollout"
    assert captured["validation_only"] is True


def test_direct_validation_replaces_dummy_load_format():
    assert direct_validation_load_format("dummy") == "auto"
    assert direct_validation_load_format("dummy_dtensor") == "auto"
    assert direct_validation_load_format("safetensors") == "safetensors"


def test_sync_hyper_generation_keeps_private_gold_metadata():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.async_rollout_mode = False
    trainer.config = OmegaConf.create({"hyper_r1": {"enable": True}})
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "position_ids": torch.tensor([[0, 1]]),
        },
        non_tensors={
            "prompt": np.array([[{"role": "user", "content": "question"}]], dtype=object),
            "reward_model": np.array([{"ground_truth": {"function_list": ["x"]}}], dtype=object),
            "extra_info": np.array([{"id": "q1"}], dtype=object),
        },
    )

    generated = trainer._get_gen_batch(batch)

    assert generated.non_tensor_batch["reward_model"][0]["ground_truth"]["function_list"] == ["x"]
    assert generated.non_tensor_batch["extra_info"][0]["id"] == "q1"
