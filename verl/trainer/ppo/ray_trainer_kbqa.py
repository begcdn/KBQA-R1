# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

# KBQA-R1 S-Expression support imports
from kbqa_r1.llm_agent.sexpr_generation import (SExprGenerationConfig,
                                                SExprLLMGenerationManager)
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           process_validation_metrics)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import (Role, WorkerType, need_critic,
                                    need_reference_policy, need_reward_model)
from verl.utils.checkpoint.checkpoint_manager import (find_latest_ckpt_path,
                                                      should_save_ckpt_esi)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (get_seqlen_balanced_partitions,
                                         log_seqlen_unbalance)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

# Initialize module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(os.getenv("VERL_LOG_LEVEL", "INFO"))
torch.autograd.set_detect_anomaly(True)


def _validation_metadata_id(metadata):
    if isinstance(metadata, Mapping):
        value = metadata.get("id")
        return str(value) if value is not None else None
    return None


def _load_completed_validation_ids(path):
    completed = set()
    if not os.path.isfile(path):
        return completed
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = _validation_metadata_id(row.get("metadata"))
            if sample_id:
                completed.add(sample_id)
    return completed


def _coerce_action_records(records):
    """Return Ray's per-sample action records as an ordinary Python list."""
    if records is None:
        return []
    return list(records)


def _normalize_generated_batch_dtypes(batch):
    """Normalize integral generation fields without destroying score precision."""
    for key, value in list(batch.items()):
        if not torch.is_floating_point(value):
            batch[key] = value.long()
    return batch

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        # Fork-R1 keeps standard outcome GRPO for the trajectory, then adds
        # paired intervention credit only to the factual graph-action tokens.
        fork_fields = {
            "fork_r1_action_mask",
            "fork_r1_factual_reward",
            "fork_r1_counterfactual_reward",
        }
        fork_enabled = bool(config.get("fork_r1_enable", False))
        missing_fork_fields = fork_fields.difference(data.batch.keys())
        if fork_enabled and missing_fork_fields:
            raise RuntimeError(
                "Fork-R1 is enabled but counterfactual rollout fields are missing: "
                + ", ".join(sorted(missing_fork_fields))
            )
        if not missing_fork_fields:
            from kbqa_r1.fork_r1 import apply_counterfactual_credit

            weight = float(config.get("fork_r1_credit_weight", 1.0))
            data.batch["advantages"] = apply_counterfactual_credit(
                data.batch["advantages"],
                data.batch["fork_r1_action_mask"],
                data.batch["fork_r1_factual_reward"],
                data.batch["fork_r1_counterfactual_reward"],
                weight=weight,
            )
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    hyper_enabled = bool(config.get("hyper_r1_enable", False))
    if hyper_enabled:
        required = {
            "hyper_r1_action_ids",
            "hyper_r1_invalid_action_mask",
            "hyper_r1_forced_terminal",
            "terminal_reward",
        }
        missing = required.difference(data.batch.keys())
        if "hyper_r1_action_records" not in data.non_tensor_batch:
            missing.add("hyper_r1_action_records")
        if missing:
            raise RuntimeError(
                "HyPER-R1 decision-credit fields are missing: "
                + ", ".join(sorted(missing))
            )
        from kbqa_r1.hyper_r1 import (
            apply_grouped_decision_credit,
            penalize_invalid_actions,
        )

        deliberate = ~data.batch["hyper_r1_forced_terminal"].to(dtype=torch.bool)

        advantages, compared_mask = apply_grouped_decision_credit(
            data.batch["advantages"],
            data.batch["hyper_r1_action_ids"],
            data.batch["terminal_reward"],
            data.non_tensor_batch["uid"],
            data.non_tensor_batch["hyper_r1_action_records"],
            weight=float(config.get("hyper_r1_credit_weight", 1.0)),
            eligible_rollouts=deliberate,
        )
        data.batch["advantages"] = advantages
        data.batch["hyper_r1_compared_action_mask"] = compared_mask
        data.batch["advantages"] = penalize_invalid_actions(
            data.batch["advantages"],
            data.batch["hyper_r1_invalid_action_mask"],
            penalty=float(config.get("hyper_r1_invalid_action_penalty", 0.25)),
        )
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        structural_constraints = self.config.get("hyper_r1", {}).get(
            "structural_constraints", False
        )
        if structural_constraints:
            if not self.config.get("hyper_r1", {}).get("enable", False):
                raise ValueError(
                    "HyPER structural constraints require hyper_r1.enable=true"
                )
            if self.config.actor_rollout_ref.rollout.name != "vllm":
                raise ValueError(
                    "HyPER structural constraints currently require the vLLM rollout"
                )
            if self.config.actor_rollout_ref.rollout.mode == "async":
                raise ValueError(
                    "HyPER structural constraints do not support async rollout"
                )
            actor_strategy = self.config.actor_rollout_ref.actor.strategy
            if actor_strategy not in ("fsdp", "fsdp2"):
                raise ValueError(
                    "HyPER structural constraints currently require an FSDP actor"
                )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        # max_val_samples = getattr(self.config.trainer, 'max_val_samples', None)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        dump_path,
        metadata=None,
        decision_traces=None,
        filename=None,
        append=False,
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, filename or f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        if metadata is not None:
            if len(metadata) != n:
                raise ValueError("generation metadata must align with dumped samples")
            base_data["metadata"] = metadata
        if decision_traces is not None:
            if len(decision_traces) != n:
                raise ValueError("decision traces must align with dumped samples")
            base_data["hyper_r1_decision_trace"] = decision_traces

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            trace = entry.pop("hyper_r1_decision_trace", None)
            if trace is not None:
                if not isinstance(trace, Mapping):
                    raise ValueError("HyPER decision trace must be a mapping")
                entry.update(dict(trace))
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "a" if append else "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
                decision_traces=batch.non_tensor_batch.get(
                    "hyper_r1_decision_trace"
                ),
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # HyPER's terminal certificate needs private gold metadata, but the
        # generation manager never serializes these fields into model input.
        # Keep the same metadata handoff used by the async agent loop.
        if self.async_rollout_mode or self.config.get("hyper_r1", {}).get("enable", False):
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _create_generation_manager_for_sexpr_mode(self, is_validation=False):
        """Create generation manager based on S-Expression mode configuration.
        
        Args:
            is_validation: Whether this is for validation (affects certain settings)
            
        Returns:
            Either SExprLLMGenerationManager or LLMGenerationManager based on config
        """
            
        enable_sexpr_mode = self.config.get('sexpr_config', {}).get('enable_sexpr_mode', False)
        
        if not enable_sexpr_mode:
            return None
            
        mode_str = "VALIDATION" if is_validation else "TRAINING"
        print(f"[VERL-{mode_str}] 🚀 EXPERIMENT: {self.config.trainer.experiment_name} - Step {self.global_steps}")
        print(f"[VERL-{mode_str}] ✓ S-Expression mode ENABLED - Using SExprLLMGenerationManager")
        print(f"[VERL-{mode_str}] ✓ Config: action_reasoning={self.config.get('sexpr_config', {}).get('enable_action_reasoning', True)}, semantic_validation={self.config.get('sexpr_config', {}).get('enable_semantic_validation', True)}")
        
        # Use S-Expression Generation Manager
        gen_config = SExprGenerationConfig(
            max_turns=self.config.get('max_turns', 5),
            max_start_length=self.config.data.get('max_start_length', 256),
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.get('max_obs_length', 512),
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.get('no_think_rl', False),
            enable_sexpr_mode=True,
            enable_action_validation=self.config.get('sexpr_config', {}).get('enable_action_reasoning', True),
            enable_sexpr_validation=self.config.get('sexpr_config', {}).get('enable_semantic_validation', True),
            hyper_r1_enable=self.config.get('hyper_r1', {}).get('enable', False),
            hyper_r1_structural_constraints=self.config.get('hyper_r1', {}).get('structural_constraints', False),
            hyper_r1_max_active=self.config.get('hyper_r1', {}).get('max_active', 24),
            hyper_r1_max_nodes=self.config.get('hyper_r1', {}).get('max_nodes', 128),
            hyper_r1_max_execution_attempts=self.config.get('hyper_r1', {}).get('max_execution_attempts', 24),
            hyper_r1_frontier_width=self.config.get('hyper_r1', {}).get('frontier_width', 6),
            hyper_r1_relation_model=self.config.get('hyper_r1', {}).get('relation_model'),
            hyper_r1_relation_device=self.config.get('hyper_r1', {}).get('relation_device'),
            sparql_url=self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
            use_odbc=self.config.get('use_odbc', False),
            use_aioodbc=self.config.get('use_aioodbc', True),
            odbc_config=self.config.get('odbc_config', None),
            experiment_name=self.config.trainer.experiment_name,
            current_step=self.global_steps
        )
        
        # Enhance odbc_config with experiment tracking information
        enhanced_odbc_config = None
        if self.config.get('odbc_config'):
            enhanced_odbc_config = dict(self.config.get('odbc_config'))
            enhanced_odbc_config['experiment_name'] = self.config.trainer.experiment_name
            enhanced_odbc_config['current_step'] = self.global_steps
        
        generation_manager = SExprLLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=is_validation,
            sparql_config={
                'sparql_url': self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                'sparql_batch_size': self.config.get('sparql_batch_size', 128),
                'sparql_max_concurrent': self.config.get('sparql_max_concurrent', 16),
                'use_odbc': self.config.get('use_odbc', False),
                'use_aioodbc': self.config.get('use_aioodbc', True),
                'odbc_config': enhanced_odbc_config
            }
        )
        
        return generation_manager

    def _validate(self):
        """Run validation loop with optional tqdm progress bar.

        在 KBQA-R1 的 rejection_sampling / val_only 场景下，验证阶段可能非常长，
        默认的 Ray/verl 日志又比较稀疏，因此这里加入一个基于样本数的进度统计：

        - 支持通过 config.trainer.max_val_samples 限制最多验证多少条样本
        - 使用 tqdm 在控制台显示 "VAL x / total (xx%)" 风格的进度条
        - 如果 tqdm 不可用或 stdout 不是 TTY，则退化为每个 batch 简单打印一次统计信息
        """

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_metadata = []
        sample_decision_traces = []
        max_val_samples = getattr(self.config.trainer, "max_val_samples", None)
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        incremental_dump = bool(self.config.trainer.get("incremental_validation_dump", False))
        progress_path = os.path.join(val_data_dir, "progress.jsonl") if val_data_dir else None
        completed_ids = (
            _load_completed_validation_ids(progress_path)
            if incremental_dump and progress_path
            else set()
        )

        # 统计总的 validation 样本数，用于 tqdm total
        # 1) 如果用户显式设置了 max_val_samples，则以此为 total
        # 2) 否则使用 len(self.val_dataset) 作为 total
        try:
            total_val_samples = int(max_val_samples) if max_val_samples is not None else len(self.val_dataset)
        except Exception:
            total_val_samples = None

        processed_samples = len(completed_ids)

        # 仅在 driver 上显示 tqdm；val_only=true 时也生效
        use_tqdm = total_val_samples is not None
        val_pbar = None
        if use_tqdm:
            # desc 中加上 experiment_name 方便区分多任务
            desc = f"Validate ({self.config.trainer.experiment_name})"
            try:
                val_pbar = tqdm(total=total_val_samples, desc=desc, ncols=120)
            except Exception:
                # tqdm 初始化失败时退化为纯 logger 模式
                module_logger.warning("[VERL-VAL] tqdm initialization failed; falling back to logger-only progress.")
                val_pbar = None
            
        for test_data in self.val_dataloader:
            # Respect validation sample cap if configured
            if max_val_samples and processed_samples >= max_val_samples:
                module_logger.info(
                    "[VERL-VAL] Reached validation sample limit: %d (processed=%d)",
                    max_val_samples,
                    processed_samples,
                )
                break
            test_batch = DataProto.from_single_dict(test_data)

            if completed_ids:
                metadata_values = test_batch.non_tensor_batch.get("extra_info")
                if metadata_values is not None:
                    keep_indices = [
                        index
                        for index, metadata_value in enumerate(metadata_values)
                        if _validation_metadata_id(metadata_value) not in completed_ids
                    ]
                    if not keep_indices:
                        continue
                    test_batch = test_batch[keep_indices]

            batch_dump_start = len(sample_scores)

            # Correctly compute original batch size (before repeat). Using len(test_data)
            # would count dict keys instead of samples, causing the cap to be ignored.
            try:
                # Prefer tensor batch when available
                original_batch_size = (
                    test_batch.batch["input_ids"].shape[0]
                    if (test_batch.batch is not None and "input_ids" in test_batch.batch.keys())
                    else (len(test_data["input_ids"]) if isinstance(test_data, dict) and "input_ids" in test_data else len(test_batch))
                )
            except Exception:
                # Fallback to DataProto length
                original_batch_size = len(test_batch)

            # If we only want a subset to honor max_val_samples, slice before any repeats
            if max_val_samples:
                remaining = int(max_val_samples - processed_samples)
                if remaining <= 0:
                    print(f"[VERL-VAL] Reached validation sample limit: {max_val_samples}")
                    break
                if original_batch_size > remaining:
                    # Slice to remaining samples to avoid processing the whole big batch
                    test_batch = test_batch[:remaining]
                    original_batch_size = remaining

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            metadata = test_batch.non_tensor_batch.get("extra_info")
            if metadata is None:
                sample_metadata.extend({} for _ in range(len(test_batch)))
            else:
                sample_metadata.extend(
                    dict(value) if isinstance(value, dict) else {}
                    for value in metadata
                )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            
            # Check if S-Expression search mode is enabled (do_search)
            enable_do_search = self.config.get('do_search', False)
            
            if not enable_do_search:
                # Original verl_newest logic: simple generation
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            else:
                # KBQA-R1 S-Expression mode: use LLMGenerationManager with run_llm_loop
                print(f"[VERL-VAL] 🚀 EXPERIMENT: {self.config.trainer.experiment_name} - Step {self.global_steps}")
                print("[VERL-VAL] 🔍 S-Expression search mode (do_search=True) - Using run_llm_loop")
                
                # Initialize generation manager for validation
                enable_sexpr_mode = self.config.get('sexpr_config', {}).get('enable_sexpr_mode', False)


                from kbqa_r1.llm_agent.sexpr_generation import \
                    SExprGenerationConfig

                # Get config values with safe defaults for KBQA-specific fields
                max_start_length = self.config.data.get('max_start_length', self.config.data.max_prompt_length // 2)
                max_obs_length = self.config.data.get('max_obs_length', 512)
                
                gen_config = SExprGenerationConfig(
                    max_turns=self.config.get('max_turns', 10),
                    max_start_length=max_start_length,
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    max_obs_length=max_obs_length,
                    num_gpus=self.config.trainer.n_gpus_per_node,
                    no_think_rl=self.config.algorithm.get('no_think_rl', False),
                    enable_sexpr_mode=True,
                    enable_action_validation=self.config.get('sexpr_config', {}).get('enable_action_reasoning', True),
                    enable_sexpr_validation=self.config.get('sexpr_config', {}).get('enable_semantic_validation', True),
                    hyper_r1_enable=self.config.get('hyper_r1', {}).get('enable', False),
                    hyper_r1_structural_constraints=self.config.get('hyper_r1', {}).get('structural_constraints', False),
                    hyper_r1_max_active=self.config.get('hyper_r1', {}).get('max_active', 24),
                    hyper_r1_max_nodes=self.config.get('hyper_r1', {}).get('max_nodes', 128),
                    hyper_r1_max_execution_attempts=self.config.get('hyper_r1', {}).get('max_execution_attempts', 24),
                    hyper_r1_frontier_width=self.config.get('hyper_r1', {}).get('frontier_width', 6),
                    hyper_r1_relation_model=self.config.get('hyper_r1', {}).get('relation_model'),
                    hyper_r1_relation_device=self.config.get('hyper_r1', {}).get('relation_device'),
                    sparql_url=self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                    use_odbc=self.config.get('use_odbc', False),
                    use_aioodbc=self.config.get('use_aioodbc', True),
                    odbc_config=self.config.get('odbc_config', None),
                    experiment_name=self.config.trainer.experiment_name,
                    current_step=self.global_steps
                )
                
                # Enhance odbc_config with experiment tracking information
                enhanced_odbc_config = None
                if self.config.get('odbc_config'):
                    enhanced_odbc_config = dict(self.config.get('odbc_config'))
                    enhanced_odbc_config['experiment_name'] = self.config.trainer.experiment_name
                    enhanced_odbc_config['current_step'] = self.global_steps
                
                generation_manager = SExprLLMGenerationManager(
                    tokenizer=self.tokenizer,
                    actor_rollout_wg=self.actor_rollout_wg,
                    config=gen_config,
                    is_validation=True,
                    sparql_config={
                        'sparql_url': self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                        'sparql_batch_size': self.config.get('sparql_batch_size', 128),
                        'sparql_max_concurrent': self.config.get('sparql_max_concurrent', 16),
                        'use_odbc': self.config.get('use_odbc', False),
                        'use_aioodbc': self.config.get('use_aioodbc', True),
                        'odbc_config': enhanced_odbc_config
                    }
                )


                
                # Run LLM loop with generation manager (process padded batch)
                first_input_ids = test_gen_batch_padded.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                
                # Update current step for logging
                generation_manager.config.current_step = self.global_steps
                
                # Update ODBC config current_step for SPARQL logging
                if hasattr(generation_manager, 'sparql_manager') and generation_manager.sparql_manager.config.odbc_config:
                    generation_manager.sparql_manager.config.odbc_config['current_step'] = self.global_steps
                elif hasattr(generation_manager, 'sexpr_executor') and hasattr(generation_manager.sexpr_executor, 'sparql_manager') and generation_manager.sexpr_executor.sparql_manager.config.odbc_config:
                    generation_manager.sexpr_executor.sparql_manager.config.odbc_config['current_step'] = self.global_steps
                
                test_output_gen_batch_padded = generation_manager.run_llm_loop(
                    gen_batch=test_gen_batch_padded,
                    initial_input_ids=first_input_ids,
                )
                
                # Token and mask tensors are integral, but HyPER's committed-answer
                # F1 and other rollout statistics must retain fractional values.
                _normalize_generated_batch_dtypes(test_output_gen_batch_padded.batch)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            module_logger.info("[VERL-VAL] Validation generation end for current batch.")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            traces = test_batch.non_tensor_batch.get("hyper_r1_decision_trace")
            if traces is not None:
                sample_decision_traces.extend(traces.tolist())

            if "hyper_r1_execution_counts" in test_batch.batch:
                execution_counts = (
                    test_batch.batch["hyper_r1_execution_counts"].detach().cpu().tolist()
                )
                reward_extra_infos_dict["hyper_r1_execution_attempts"].extend(
                    float(value) for value in execution_counts
                )
            if "hyper_r1_premature_answer" in test_batch.batch:
                reward_extra_infos_dict["hyper_r1_premature_answer"].extend(
                    float(value)
                    for value in test_batch.batch["hyper_r1_premature_answer"]
                    .detach()
                    .cpu()
                    .tolist()
                )
            if "hyper_r1_action_records" in test_batch.non_tensor_batch:
                for records in test_batch.non_tensor_batch["hyper_r1_action_records"]:
                    records = _coerce_action_records(records)
                    selected = [
                        str(record.get("node_id", ""))
                        for record in records
                        if record.get("action") == "Select"
                    ]
                    reward_extra_infos_dict["hyper_r1_branch_switch"].append(
                        float(len(set(selected)) >= 2)
                    )
                    reward_extra_infos_dict["hyper_r1_used_combine"].append(
                        float(any(record.get("action") == "Combine" for record in records))
                    )
                    reward_extra_infos_dict["hyper_r1_used_widen"].append(
                        float(any(record.get("action") == "Widen" for record in records))
                    )
                    reward_extra_infos_dict["hyper_r1_max_active"].append(
                        float(max((record.get("active_before", 0) for record in records), default=0))
                    )
                    reward_extra_infos_dict["hyper_r1_preserved_alternatives"].append(
                        float(any(record.get("active_before", 0) >= 2 for record in records))
                    )

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if self.config.algorithm.get("hyper_r1_enable", False):
                required_hyper_fields = {
                    "hyper_r1_commit_valid",
                    "hyper_r1_commit_protocol_valid",
                    "hyper_r1_commit_answer_exact",
                    "hyper_r1_commit_answer_f1",
                    "hyper_r1_commit_intent_equivalent",
                    "hyper_r1_abstained",
                    "hyper_r1_explicit_model_commit",
                    "hyper_r1_forced_terminal",
                    "hyper_r1_forced_empty",
                    "hyper_r1_turn_exhausted",
                }
                missing_hyper_fields = required_hyper_fields.difference(
                    test_batch.batch.keys()
                )
                if missing_hyper_fields:
                    raise RuntimeError(
                        "HyPER-R1 validation is missing fields: "
                        + ", ".join(sorted(missing_hyper_fields))
                    )
                from kbqa_r1.hyper_r1 import enforce_commit_reward

                validation_mask = compute_response_mask(test_batch)
                reward_tensor = enforce_commit_reward(
                    reward_tensor,
                    validation_mask,
                    test_batch.batch["hyper_r1_commit_protocol_valid"],
                    test_batch.batch["hyper_r1_commit_answer_f1"],
                    commit_intent_equivalent=test_batch.batch[
                        "hyper_r1_commit_intent_equivalent"
                    ],
                    abstained=test_batch.batch["hyper_r1_abstained"],
                    forced_terminal=test_batch.batch["hyper_r1_forced_terminal"],
                    invalid_penalty=float(
                        self.config.algorithm.get(
                            "hyper_r1_invalid_commit_penalty", 0.25
                        )
                    ),
                    semantic_bonus=float(
                        self.config.algorithm.get("hyper_r1_semantic_bonus", 0.1)
                    ),
                    forced_terminal_penalty=float(
                        self.config.algorithm.get(
                            "hyper_r1_forced_terminal_penalty", 0.25
                        )
                    ),
                )
                reward_extra_infos_dict["hyper_r1_commit_valid"].extend(
                    float(value)
                    for value in test_batch.batch["hyper_r1_commit_valid"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                reward_extra_infos_dict["hyper_r1_commit_protocol_valid"].extend(
                    float(value)
                    for value in test_batch.batch[
                        "hyper_r1_commit_protocol_valid"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                reward_extra_infos_dict["hyper_r1_commit_answer_exact"].extend(
                    float(value)
                    for value in test_batch.batch["hyper_r1_commit_answer_exact"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                reward_extra_infos_dict["hyper_r1_commit_answer_f1"].extend(
                    float(value)
                    for value in test_batch.batch["hyper_r1_commit_answer_f1"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                reward_extra_infos_dict["hyper_r1_commit_intent_equivalent"].extend(
                    float(value)
                    for value in test_batch.batch[
                        "hyper_r1_commit_intent_equivalent"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                reward_extra_infos_dict["hyper_r1_abstained"].extend(
                    float(value)
                    for value in test_batch.batch["hyper_r1_abstained"]
                    .detach()
                    .cpu()
                    .tolist()
                )
                for field in (
                    "hyper_r1_explicit_model_commit",
                    "hyper_r1_forced_terminal",
                    "hyper_r1_forced_empty",
                    "hyper_r1_turn_exhausted",
                ):
                    reward_extra_infos_dict[field].extend(
                        float(value)
                        for value in test_batch.batch[field]
                        .detach()
                        .cpu()
                        .tolist()
                    )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            module_logger.debug(
                "[VERL-VAL] len(reward_extra_infos_dict['reward'])=%d",
                len(reward_extra_infos_dict["reward"]),
            )
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    module_logger.debug(
                        "[VERL-VAL] len(reward_extra_infos_dict['%s'])=%d",
                        key,
                        len(reward_extra_infos_dict[key]),
                    )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            if incremental_dump and val_data_dir:
                batch_dump_end = len(sample_scores)
                batch_extra_infos = {
                    key: values[batch_dump_start:batch_dump_end]
                    for key, values in reward_extra_infos_dict.items()
                    if len(values) >= batch_dump_end
                }
                self._dump_generations(
                    inputs=sample_inputs[batch_dump_start:batch_dump_end],
                    outputs=sample_outputs[batch_dump_start:batch_dump_end],
                    gts=sample_gts[batch_dump_start:batch_dump_end],
                    scores=sample_scores[batch_dump_start:batch_dump_end],
                    reward_extra_infos_dict=batch_extra_infos,
                    dump_path=val_data_dir,
                    metadata=sample_metadata[batch_dump_start:batch_dump_end],
                    decision_traces=(
                        sample_decision_traces[batch_dump_start:batch_dump_end]
                        if sample_decision_traces
                        else None
                    ),
                    filename="progress.jsonl",
                    append=True,
                )

            # Increment processed samples by the true number of samples in this batch (before repeat)
            processed_samples += int(original_batch_size)

            # 更新 tqdm 进度条（按样本数，而不是 batch 数）
            if val_pbar is not None:
                try:
                    val_pbar.update(int(original_batch_size))
                    if total_val_samples is not None:
                        val_pbar.set_postfix({"processed": processed_samples, "total": total_val_samples})
                except Exception:
                    # tqdm 出错时不影响主流程
                    pass
            else:
                # 没有 tqdm 时，退化为简单的进度日志
                if total_val_samples is not None:
                    fraction = processed_samples / max(total_val_samples, 1)
                    module_logger.info(
                        "[VERL-VAL] progress: %d/%d (%.1f%%)",
                        processed_samples,
                        total_val_samples,
                        fraction * 100.0,
                    )
                else:
                    module_logger.info("[VERL-VAL] processed %d samples", processed_samples)

            if max_val_samples and processed_samples >= max_val_samples:
                module_logger.info(
                    "[VERL-VAL] Reached validation sample limit: %d (processed=%d)",
                    max_val_samples,
                    processed_samples,
                )
                break

        # 关闭 tqdm 进度条
        if val_pbar is not None:
            try:
                val_pbar.close()
            except Exception:
                pass

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                metadata=sample_metadata,
                decision_traces=(sample_decision_traces or None),
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            validation_only = self.config.trainer.get("val_only", False)
            worker_role = "rollout" if validation_only else "actor_rollout"
            worker_config = self.config.actor_rollout_ref
            if validation_only:
                with open_dict(worker_config):
                    worker_config.validation_only = True
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=worker_config,
                role=worker_role,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Check if S-Expression search mode is enabled (do_search)
                    enable_do_search = self.config.get('do_search', False)
                    
                    if not enable_do_search:
                        # Original verl_newest logic: simple generation
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                    else:
                        # KBQA-R1 S-Expression mode: use LLMGenerationManager with run_llm_loop
                        print(f"[VERL-TRAIN] 🚀 EXPERIMENT: {self.config.trainer.experiment_name} - Step {self.global_steps}")
                        print("[VERL-TRAIN] 🔍 S-Expression search mode (do_search=True) - Using run_llm_loop")
                        
                        # Initialize generation manager (created once per step to update config)
                        enable_sexpr_mode = self.config.get('sexpr_config', {}).get('enable_sexpr_mode', False)
                        
                        if enable_sexpr_mode:
                            print("[VERL-TRAIN] ✓ S-Expression mode ENABLED - Using SExprLLMGenerationManager")
                            print(f"[VERL-TRAIN] ✓ Config: action_reasoning={self.config.get('sexpr_config', {}).get('enable_action_reasoning', True)}, semantic_validation={self.config.get('sexpr_config', {}).get('enable_semantic_validation', True)}")
                            
                            from kbqa_r1.llm_agent.sexpr_generation import \
                                SExprGenerationConfig

                            # Get config values with safe defaults for KBQA-specific fields
                            max_start_length = self.config.data.get('max_start_length', self.config.data.max_prompt_length // 2)
                            max_obs_length = self.config.data.get('max_obs_length', 512)
                            
                            gen_config = SExprGenerationConfig(
                                max_turns=self.config.get('max_turns', 10),
                                max_start_length=max_start_length,
                                max_prompt_length=self.config.data.max_prompt_length,
                                max_response_length=self.config.data.max_response_length,
                                max_obs_length=max_obs_length,
                                num_gpus=self.config.trainer.n_gpus_per_node,
                                no_think_rl=self.config.algorithm.get('no_think_rl', False),
                                enable_sexpr_mode=True,
                                enable_action_validation=self.config.get('sexpr_config', {}).get('enable_action_reasoning', True),
                                enable_sexpr_validation=self.config.get('sexpr_config', {}).get('enable_semantic_validation', True),
                                hyper_r1_enable=self.config.get('hyper_r1', {}).get('enable', False),
                                hyper_r1_structural_constraints=self.config.get('hyper_r1', {}).get('structural_constraints', False),
                                hyper_r1_max_active=self.config.get('hyper_r1', {}).get('max_active', 24),
                                hyper_r1_max_nodes=self.config.get('hyper_r1', {}).get('max_nodes', 128),
                                hyper_r1_max_execution_attempts=self.config.get('hyper_r1', {}).get('max_execution_attempts', 24),
                                hyper_r1_frontier_width=self.config.get('hyper_r1', {}).get('frontier_width', 6),
                                hyper_r1_relation_model=self.config.get('hyper_r1', {}).get('relation_model'),
                                hyper_r1_relation_device=self.config.get('hyper_r1', {}).get('relation_device'),
                                sparql_url=self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                                use_odbc=self.config.get('use_odbc', False),
                                use_aioodbc=self.config.get('use_aioodbc', True),
                                odbc_config=self.config.get('odbc_config', None),
                                experiment_name=self.config.trainer.experiment_name,
                                current_step=self.global_steps
                            )
                            
                            # Enhance odbc_config with experiment tracking information
                            enhanced_odbc_config = None
                            if self.config.get('odbc_config'):
                                enhanced_odbc_config = dict(self.config.get('odbc_config'))
                                enhanced_odbc_config['experiment_name'] = self.config.trainer.experiment_name
                                enhanced_odbc_config['current_step'] = self.global_steps
                            
                            generation_manager = SExprLLMGenerationManager(
                                tokenizer=self.tokenizer,
                                actor_rollout_wg=self.actor_rollout_wg,
                                config=gen_config,
                                sparql_config={
                                    'sparql_url': self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                                    'sparql_batch_size': self.config.get('sparql_batch_size', 128),
                                    'sparql_max_concurrent': self.config.get('sparql_max_concurrent', 16),
                                    'use_odbc': self.config.get('use_odbc', False),
                                    'use_aioodbc': self.config.get('use_aioodbc', True),
                                    'odbc_config': enhanced_odbc_config
                                }
                            )
                        # else:
                        #     print("[VERL-TRAIN] ⚠ S-Expression mode DISABLED or imports unavailable - Using original LLMGenerationManager")
                            
                        #     from kbqa_r1.llm_agent.generation import (
                        #         GenerationConfig, LLMGenerationManager)

                        #     # Get config values with safe defaults for KBQA-specific fields
                        #     max_start_length = self.config.data.get('max_start_length', self.config.data.max_prompt_length // 2)
                        #     max_obs_length = self.config.data.get('max_obs_length', 512)
                            
                        #     gen_config = GenerationConfig(
                        #         max_turns=self.config.get('max_turns', 10),
                        #         max_start_length=max_start_length,
                        #         max_prompt_length=self.config.data.max_prompt_length,
                        #         max_response_length=self.config.data.max_response_length,
                        #         max_obs_length=max_obs_length,
                        #         num_gpus=self.config.trainer.n_gpus_per_node,
                        #         no_think_rl=self.config.algorithm.get('no_think_rl', False),
                        #         sparql_url=self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                        #         use_odbc=self.config.get('use_odbc', False),
                        #         use_aioodbc=self.config.get('use_aioodbc', True),
                        #         experiment_name=self.config.trainer.experiment_name,
                        #         current_step=self.global_steps
                        #     )
                            
                        #     # Enhance odbc_config with experiment tracking information
                        #     enhanced_odbc_config = None
                        #     if self.config.get('odbc_config'):
                        #         enhanced_odbc_config = dict(self.config.get('odbc_config'))
                        #         enhanced_odbc_config['experiment_name'] = self.config.trainer.experiment_name
                        #         enhanced_odbc_config['current_step'] = self.global_steps
                            
                        #     generation_manager = LLMGenerationManager(
                        #         tokenizer=self.tokenizer,
                        #         actor_rollout_wg=self.actor_rollout_wg,
                        #         config=gen_config,
                        #         sparql_config={
                        #             'sparql_url': self.config.get('sparql', {}).get('url', 'http://localhost:8000/execute'),
                        #             'sparql_batch_size': self.config.get('sparql_batch_size', 128),
                        #             'sparql_max_concurrent': self.config.get('sparql_max_concurrent', 16),
                        #             'use_odbc': self.config.get('use_odbc', False),
                        #             'use_aioodbc': self.config.get('use_aioodbc', True),
                        #             'odbc_config': enhanced_odbc_config
                        #         }
                        #     )
                        
                        # Run LLM loop with generation manager
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                        
                        with marked_timer("gen", timing_raw, color="red"):
                            # Update current step for logging
                            generation_manager.config.current_step = self.global_steps
                            generation_manager.timing_raw = timing_raw
                            
                            # Update ODBC config current_step for SPARQL logging
                            if hasattr(generation_manager, 'sparql_manager') and generation_manager.sparql_manager.config.odbc_config:
                                generation_manager.sparql_manager.config.odbc_config['current_step'] = self.global_steps
                            elif hasattr(generation_manager, 'sexpr_executor') and hasattr(generation_manager.sexpr_executor, 'sparql_manager') and generation_manager.sexpr_executor.sparql_manager.config.odbc_config:
                                generation_manager.sexpr_executor.sparql_manager.config.odbc_config['current_step'] = self.global_steps
                            
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )
                        
                        # Convert rollout tensors to appropriate dtype (keep floats untouched)
                        for key, value in list(final_gen_batch_output.batch.items()):
                            if torch.is_floating_point(value):
                                continue
                            final_gen_batch_output.batch[key] = value.long()

                        # DEBUG: Check rollout_log_probs presence for mismatch metrics
                        if 'rollout_log_probs' in final_gen_batch_output.batch:
                            rollout_log_probs_shape = final_gen_batch_output.batch['rollout_log_probs'].shape
                            module_logger.info(
                                f"[MISMATCH DEBUG] rollout_log_probs ready for mismatch metrics, shape={rollout_log_probs_shape}"
                            )
                        else:
                            module_logger.warning("[MISMATCH DEBUG] rollout_log_probs missing before advantage computation")
                        
                        # Extract S-Expression action metrics from meta_info
                        try:
                            if hasattr(final_gen_batch_output, 'meta_info') and isinstance(final_gen_batch_output.meta_info, dict):
                                # Debug: log all sexpr-related keys in meta_info
                                sexpr_keys = [k for k in final_gen_batch_output.meta_info.keys() if isinstance(k, str) and ('sexpr' in k or 'hist/' in k)]
                                if sexpr_keys:
                                    module_logger.info(f"[SEXPR-METRICS] Found {len(sexpr_keys)} sexpr/hist keys in meta_info: {sexpr_keys[:10]}...")
                                
                                for k, v in final_gen_batch_output.meta_info.items():
                                    if isinstance(k, str):
                                        if k.startswith('sexpr/actions/'):
                                            metrics[k] = float(v)
                                        elif k.startswith('sexpr/candidate_rel_len/'):
                                            metrics[k] = float(v)
                                        elif k.startswith('sexpr/relation_similarity_top1/'):
                                            # Record TOP-1 relation similarity metrics (mean/min/max per step)
                                            metrics[k] = float(v)
                                            module_logger.info(f"[SEXPR-METRICS] Added scalar metric: {k} = {v}")
                                        elif k.startswith('turn_final/truncation/'):
                                            metrics[k] = float(v)
                                        elif k.startswith('hist/'):
                                            # pass through histogram arrays
                                            metrics[k] = v
                                            if 'relation_similarity' in k:
                                                module_logger.info(f"[SEXPR-METRICS] Added histogram: {k} with {len(v) if isinstance(v, list) else 'N/A'} values")
                        except Exception as e:
                            module_logger.warning(f"[SEXPR-METRICS] Error extracting meta_info: {e}")
                        
                        gen_batch_output = final_gen_batch_output
                        
                        # Update timing from generation manager if available
                        if hasattr(generation_manager, 'timing_raw') and generation_manager.timing_raw:
                            timing_raw.update(generation_manager.timing_raw)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if self.config.algorithm.get("hyper_r1_enable", False):
                        graph_metadata = gen_batch_output.meta_info.get("hyper_r1/graphs")
                        action_metadata = gen_batch_output.meta_info.get("hyper_r1/action_records")
                        if graph_metadata is None or action_metadata is None:
                            raise RuntimeError("HyPER-R1 rollout metadata is missing")
                        batch.non_tensor_batch["hyper_r1_graph"] = np.array(
                            [graph_metadata[i] for i in range(len(batch))], dtype=object
                        )
                        batch.non_tensor_batch["hyper_r1_action_records"] = np.array(
                            [action_metadata.get(i, []) for i in range(len(batch))], dtype=object
                        )

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import \
                                calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # HyPER-R1's task contract is part of the environment
                        # reward, not part of language-model regularization.
                        # Gate the raw task score first so invalid commits cannot
                        # earn answer reward, and retain this clean outcome for
                        # same-state sibling credit. KL is applied afterwards to
                        # every rollout, including invalid ones.
                        if self.config.algorithm.get("hyper_r1_enable", False):
                            required = {
                                "hyper_r1_execution_counts",
                                "hyper_r1_commit_valid",
                                "hyper_r1_commit_protocol_valid",
                                "hyper_r1_commit_answer_exact",
                                "hyper_r1_commit_answer_f1",
                                "hyper_r1_commit_intent_equivalent",
                                "hyper_r1_abstained",
                                "hyper_r1_forced_terminal",
                                "hyper_r1_explicit_model_commit",
                                "hyper_r1_forced_empty",
                                "hyper_r1_turn_exhausted",
                                "hyper_r1_premature_answer",
                                "response_mask",
                            }
                            missing = required.difference(batch.batch.keys())
                            if missing:
                                raise RuntimeError(
                                    "HyPER-R1 rollout fields are missing: "
                                    + ", ".join(sorted(missing))
                                )
                            from kbqa_r1.hyper_r1 import (
                                charge_execution_budget,
                                enforce_commit_reward,
                            )

                            hyper_task_rewards = enforce_commit_reward(
                                batch.batch["token_level_scores"],
                                batch.batch["response_mask"],
                                batch.batch["hyper_r1_commit_protocol_valid"],
                                batch.batch["hyper_r1_commit_answer_f1"],
                                commit_intent_equivalent=batch.batch[
                                    "hyper_r1_commit_intent_equivalent"
                                ],
                                abstained=batch.batch["hyper_r1_abstained"],
                                forced_terminal=batch.batch[
                                    "hyper_r1_forced_terminal"
                                ],
                                invalid_penalty=float(
                                    self.config.algorithm.get(
                                        "hyper_r1_invalid_commit_penalty", 0.25
                                    )
                                ),
                                semantic_bonus=float(
                                    self.config.algorithm.get(
                                        "hyper_r1_semantic_bonus", 0.0
                                    )
                                ),
                                forced_terminal_penalty=float(
                                    self.config.algorithm.get(
                                        "hyper_r1_forced_terminal_penalty", 0.25
                                    )
                                ),
                            )
                            hyper_task_rewards = charge_execution_budget(
                                hyper_task_rewards,
                                batch.batch["response_mask"],
                                batch.batch["hyper_r1_execution_counts"],
                                max_execution_attempts=int(
                                    self.config.get("hyper_r1", {}).get(
                                        "max_execution_attempts", 24
                                    )
                                ),
                                cost=float(
                                    self.config.algorithm.get("hyper_r1_budget_cost", 0.0)
                                ),
                                group_ids=batch.non_tensor_batch["uid"],
                            )
                            batch.batch["token_level_scores"] = hyper_task_rewards
                            batch.batch["terminal_reward"] = hyper_task_rewards.sum(dim=-1)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if self.config.algorithm.get("hyper_r1_enable", False):
                            metrics["hyper_r1/commit_valid_rate"] = float(
                                batch.batch["hyper_r1_commit_valid"].float().mean().item()
                            )
                            metrics["hyper_r1/commit_protocol_valid_rate"] = float(
                                batch.batch["hyper_r1_commit_protocol_valid"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/commit_answer_exact_rate"] = float(
                                batch.batch["hyper_r1_commit_answer_exact"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/commit_answer_f1"] = float(
                                batch.batch["hyper_r1_commit_answer_f1"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/commit_intent_equivalent_rate"] = float(
                                batch.batch["hyper_r1_commit_intent_equivalent"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/abstain_rate"] = float(
                                batch.batch["hyper_r1_abstained"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/explicit_model_commit_rate"] = float(
                                batch.batch["hyper_r1_explicit_model_commit"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/forced_terminal_rate"] = float(
                                batch.batch["hyper_r1_forced_terminal"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/forced_empty_rate"] = float(
                                batch.batch["hyper_r1_forced_empty"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/turn_exhaustion_rate"] = float(
                                batch.batch["hyper_r1_turn_exhausted"]
                                .float()
                                .mean()
                                .item()
                            )
                            metrics["hyper_r1/mean_execution_attempts"] = float(
                                batch.batch["hyper_r1_execution_counts"].float().mean().item()
                            )
                            metrics["hyper_r1/premature_answer_rate"] = float(
                                batch.batch["hyper_r1_premature_answer"]
                                .float()
                                .mean()
                                .item()
                            )
                            action_tokens = batch.batch["hyper_r1_action_ids"] > 0
                            compared_tokens = batch.batch[
                                "hyper_r1_compared_action_mask"
                            ] > 0
                            metrics["hyper_r1/action_token_rate"] = float(
                                action_tokens.float().mean().item()
                            )
                            metrics["hyper_r1/compared_action_token_rate"] = float(
                                compared_tokens.sum().item()
                                / max(1, action_tokens.sum().item())
                            )
                            metrics["hyper_r1/invalid_policy_token_rate"] = float(
                                batch.batch["hyper_r1_invalid_action_mask"]
                                .float()
                                .mean()
                                .item()
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            # Apply state masking for S-Expression search mode if enabled
                            if self.config.get('do_search', False) and self.config.actor_rollout_ref.actor.get('state_masking', False):
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # [DEBUG] Optionally dump samples contributing to response_length/min
                try:
                    import os as _os
                    debug_flag = (_os.getenv('VERL_DEBUG_RESP_LEN', '0') == '1') or bool(
                        getattr(getattr(self.config, 'trainer', object()), 'debug_response_length_min', False)
                    )
                    if debug_flag and 'responses' in batch.batch and 'attention_mask' in batch.batch:
                        responses = batch.batch['responses']
                        resp_len_total = int(responses.shape[-1])
                        attn_mask = batch.batch['attention_mask']
                        prompt_mask = attn_mask[:, :-resp_len_total]
                        response_mask = attn_mask[:, -resp_len_total:]
                        prompt_length = prompt_mask.sum(-1).float()
                        response_length = response_mask.sum(-1).float()

                        min_len_val = float(torch.min(response_length).item())
                        min_indices = (response_length == min_len_val).nonzero(as_tuple=False).view(-1).tolist()
                        if len(min_indices) > 0:
                            print('[VERL][DEBUG] response_length/min samples (batch) -> min=', int(min_len_val))

                            # Resolve output path for JSONL logging
                            try:
                                import json as _json
                                import time as _time
                                file_path = _os.getenv('VERL_DEBUG_RESP_LEN_FILE', None)
                                if not file_path:
                                    base_dir = getattr(self.config.trainer, 'default_local_dir', '.')
                                    cfg_file = getattr(self.config.trainer, 'debug_response_length_min_file', None)
                                    file_path = cfg_file or _os.path.join(base_dir, 'response_len_min.jsonl')
                                parent_dir = _os.path.dirname(file_path)
                                if parent_dir:
                                    _os.makedirs(parent_dir, exist_ok=True)
                            except Exception:
                                file_path = None

                            for idx in min_indices:
                                try:
                                    uid = None
                                    if 'uid' in batch.non_tensor_batch:
                                        uid = batch.non_tensor_batch['uid'][idx]
                                    p_len = int(prompt_length[idx].item())
                                    r_len = int(response_length[idx].item())
                                    prompt_text = None
                                    if 'input_ids' in batch.batch:
                                        prompt_ids = batch.batch['input_ids'][idx][:p_len]
                                        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                                    resp_ids = batch.batch['responses'][idx][:r_len]
                                    resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                                    data_source = None
                                    if 'data_source' in batch.non_tensor_batch:
                                        data_source = batch.non_tensor_batch['data_source'][idx]
                                    print('-- idx=', idx, 'uid=', uid, 'data_source=', data_source)
                                    if prompt_text is not None:
                                        print('   prompt(', p_len, '):', prompt_text)
                                    print('   response(', r_len, '):', resp_text)

                                    # Append to JSONL file
                                    try:
                                        if file_path:
                                            record = {
                                                'experiment': getattr(self.config.trainer, 'experiment_name', None),
                                                'epoch': int(epoch),
                                                'step': int(self.global_steps),
                                                'min_response_len': int(min_len_val),
                                                'batch_index': int(idx),
                                                'uid': str(uid) if uid is not None else None,
                                                'data_source': str(data_source) if data_source is not None else None,
                                                'prompt_len': p_len,
                                                'response_len': r_len,
                                                'prompt': prompt_text,
                                                'response': resp_text,
                                                'ts': _time.time(),
                                            }
                                            with open(file_path, 'a', encoding='utf-8') as f:
                                                f.write(_json.dumps(record, ensure_ascii=False) + '\n')
                                    except Exception as _werr:
                                        print('[VERL][DEBUG] Failed to write min sample to file:', _werr)
                                except Exception as _e:
                                    print('[VERL][DEBUG] Failed to print min sample at idx', idx, 'error:', _e)
                except Exception:
                    # Never break training due to debug printing
                    pass
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens in KBQA S-Expression mode.
        
        This method masks certain tokens during loss computation based on either:
        1. Pre-computed info_mask in the batch (preferred)
        2. State markers in the response text (fallback)
        
        Args:
            batch: DataProto containing responses and attention masks
            metrics: Dictionary to update with masking statistics
            
        Returns:
            Tuple of (batch with loss_mask added, updated metrics)
        """
        response_length = batch.batch['responses'].shape[-1]
        response_mask = (batch.batch['info_mask'] if 'info_mask' in batch.batch 
                        else batch.batch['attention_mask'])[:, -response_length:]
        
        # If info_mask is present, use it directly as loss mask on responses
        if 'info_mask' in batch.batch:
            loss_mask = response_mask
            batch.batch['loss_mask'] = loss_mask
            batch.batch['response_mask'] = loss_mask  # Ensure response_mask is set
            metrics.update({
                'state_tokens/total': loss_mask.sum().item(),
                'state_tokens/coverage': (loss_mask.sum() / batch.batch['attention_mask'][:, -response_length:].sum()).item(),
            })
            return batch, metrics
        
        # Otherwise fall back to state marker based masking
        state_mask = torch.ones_like(response_mask)
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) 
                    for resp in batch.batch['responses']]
    
        for i, response in enumerate(responses):
            # Find all pairs of start and end marker positions
            start_marker = self.config.algorithm.get('state_masking', {}).get('start_state_marker', '<information>')
            end_marker = self.config.algorithm.get('state_masking', {}).get('end_state_marker', '</information>')
            
            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                
                state_mask[i, start_token_pos:end_token_pos] = 0
        
        loss_mask = state_mask * response_mask
        batch.batch['loss_mask'] = loss_mask
        batch.batch['response_mask'] = loss_mask  # Ensure response_mask is set

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / batch.batch['attention_mask'][:, -response_length:].sum()).item(),
        })
        
        return batch, metrics
