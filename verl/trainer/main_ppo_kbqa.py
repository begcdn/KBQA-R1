# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer_kbqa as ray_trainer_kbqa is used by other mpain.
"""

import glob
import logging
import os
import re
import socket
from datetime import datetime

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer_kbqa import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type
from verl.utils.reward_score import mid_reward, qa_em

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOG_LEVEL", "INFO"))


def _select_rm_score_fn(data_source):
    """Select reward model score function based on data source."""
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    elif data_source in ['webqsp', 'grailqa', 'graphq']:
        return mid_reward.compute_mid_reward
    else:
        raise NotImplementedError(f"Unsupported data source: {data_source}")


class RewardManager():
    """The reward manager for KBQA tasks with error logging."""

    def __init__(self, tokenizer, num_examine, format_score=0.0, structure_format_score=0.0, mid_f1_weight: float = 1.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.mid_f1_weight = mid_f1_weight
        # Track rewards history for normalization
        self.rewards_history = []
        self.history_size = 100
        # Track training step for progressive reward scaling
        self.training_step = 0
        
        # Setup error logging for test_set failures
        self._setup_error_logging()

    def _setup_error_logging(self):
        """Setup logging for test_set error samples."""
        os.makedirs('logs', exist_ok=True)
        
        self.error_logger = logging.getLogger('test_set_errors')
        self.error_logger.setLevel(logging.INFO)
        
        for handler in self.error_logger.handlers[:]:
            self.error_logger.removeHandler(handler)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = f'logs/test_set_errors_{timestamp}.log'
        file_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.error_logger.addHandler(file_handler)
        self.error_logger.propagate = False
        
        print(f"[REWARD-MANAGER] Test error logging initialized: {error_log_file}")

    def _log_test_error(self, sample_idx: int, data_source: str, sequences_str: str, 
                       ground_truth: dict, base_score: float, error_type: str, 
                       error_details: str = ""):
        """Log test_set error samples with detailed information."""
        try:
            error_info = {
                'sample_index': sample_idx,
                'data_source': data_source,
                'timestamp': datetime.now().isoformat(),
                'training_step': self.training_step,
                'error_type': error_type,
                'error_details': error_details,
                'base_score': base_score,
                'ground_truth': ground_truth,
                'predicted_sequence': sequences_str[:500] + "..." if len(sequences_str) > 500 else sequences_str
            }
            
            self.error_logger.error(f"TEST_ERROR: {error_info}")
            
        except Exception as e:
            print(f"[REWARD-MANAGER] Failed to log test error: {e}")

    def _detect_error_patterns(self, sequences_str: str) -> list:
        """Detect common error patterns in the generated sequences."""
        error_patterns = []
        
        tag_patterns = ['think', 'action', 'information', 'answer']
        for tag in tag_patterns:
            opening_count = len(re.findall(f'<{tag}>', sequences_str))
            closing_count = len(re.findall(f'</{tag}>', sequences_str))
            if opening_count != closing_count:
                error_patterns.append(f"unbalanced_{tag}_tags")
        
        if '<answer></answer>' in sequences_str or '<answer> </answer>' in sequences_str:
            error_patterns.append("empty_answer")
        
        if sequences_str.strip().endswith('<') or sequences_str.strip().endswith('('):
            error_patterns.append("incomplete_sequence")
        
        if sequences_str.count('(') != sequences_str.count(')'):
            error_patterns.append("unbalanced_parentheses")
        
        if len(sequences_str.strip()) < 10:
            error_patterns.append("too_short_response")
        
        words = sequences_str.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:
                error_patterns.append("excessive_repetition")
        
        return error_patterns

    def __call__(self, data: DataProto, return_dict: bool = True, **kwargs):
        """Compute rewards for generated responses.

        Args:
            data: DataProto batch containing prompts, responses, masks, and non-tensor info.
            return_dict: When True, return a dict with keys:
                - 'reward_tensor': torch.Tensor with per-token rewards (last token used)
                - 'reward_extra_info': dict of auxiliary metrics/info
            **kwargs: Ignored extra kwargs for compatibility.

        Returns:
            torch.Tensor or dict: Depends on return_dict flag.
        """
        if 'rm_scores' in data.batch.keys():
            reward_tensor = data.batch['rm_scores']
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = []
        already_print_data_sources = {}
        mid_f1_list = []
        structure_reward_list = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            if isinstance(ground_truth, dict) and 'target' in ground_truth:
                gt_value = ground_truth['target']
            else:
                gt_value = ground_truth
            
            # error_patterns = self._detect_error_patterns(sequences_str)
            # if error_patterns:
            #     self._log_test_error(i, data_source, sequences_str, ground_truth, 0.0, 
            #                        "pattern_errors", f"Patterns: {error_patterns}")
            # logger.warning(f"Ground Truth: {gt_value}")
            # print(f"********{gt_value}")
            compute_score_fn = _select_rm_score_fn(data_source)

            base_score = compute_score_fn(sequences_str, gt_value)['total']

            enhanced_score = base_score
            all_scores.append(enhanced_score)
            self.rewards_history.append(enhanced_score)
            if len(self.rewards_history) > self.history_size:
                self.rewards_history.pop(0)
            
            reward_tensor[i, valid_response_length - 1] = enhanced_score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
        
        if all_scores:
            print(f"[REWARDS] mean: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}, "
                  f"min: {np.min(all_scores):.4f}, std: {np.std(all_scores):.4f}")

        try:
            self.last_reward_metrics = {
                'mid_f1_mean': float(np.mean(mid_f1_list)) if len(mid_f1_list) > 0 else 0.0,
                'structure_reward_mean': float(np.mean(structure_reward_list)) if len(structure_reward_list) > 0 else 0.0,
            }
        except Exception:
            self.last_reward_metrics = {'mid_f1_mean': 0.0, 'structure_reward_mean': 0.0}

        if return_dict:
            reward_extra_info = {
                'metrics': self.last_reward_metrics,
                'rewards_summary': {
                    'mean': float(np.mean(all_scores)) if len(all_scores) > 0 else 0.0,
                    'max': float(np.max(all_scores)) if len(all_scores) > 0 else 0.0,
                    'min': float(np.min(all_scores)) if len(all_scores) > 0 else 0.0,
                    'std': float(np.std(all_scores)) if len(all_scores) > 0 else 0.0,
                },
            }
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}

        return reward_tensor
    
    def update_training_step(self, step: int):
        """Update the current training step for progressive reward scaling."""
        self.training_step = step

    def generate_error_summary(self):
        """Generate a summary of logged errors for analysis."""
        try:
            error_log_files = glob.glob('logs/test_set_errors_*.log')
            if not error_log_files:
                print("[REWARD-MANAGER] No error log files found")
                return
            
            latest_log = max(error_log_files, key=os.path.getctime)
            error_counts = {}
            
            with open(latest_log, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'error_type' in line:
                        for error_type in ['pattern_errors', 'compute_error', 'empty_answer', 'incomplete_sequence']:
                            if error_type in line:
                                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            print(f"\n[REWARD-MANAGER] Error Summary from {latest_log}:")
            print("=" * 50)
            for error_type, count in error_counts.items():
                print(f"{error_type}: {count}")
            print("=" * 50)
            
        except Exception as e:
            print(f"[REWARD-MANAGER] Failed to generate error summary: {e}")


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import (ActorRolloutRefWorker,
                                                   AsyncActorRolloutRefWorker)

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import (
                ActorRolloutRefWorker, AsyncActorRolloutRefWorker)

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer_kbqa import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer_kbqa import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer_kbqa import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer_kbqa import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer_kbqa import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer_kbqa import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Training reward function with reward_kwargs (only for kbqa reward manager)
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "kbqa":
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
        else:
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0
            )
        
        # Validation reward function with optional override for val_reward_kwargs
        # If val_reward_kwargs is specified, use it; otherwise use reward_kwargs
        if reward_manager_name == "kbqa":
            val_reward_kwargs = config.reward_model.get("val_reward_kwargs", None)
            if val_reward_kwargs is None:
                val_reward_kwargs = config.reward_model.get("reward_kwargs", {})
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **val_reward_kwargs
            )
        else:
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1
            )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
