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
Generate responses for KBQA tasks with S-Expression support and Reward computation
Lightweight rejection sampling without training overhead
"""

import json
import os
import re
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from kbqa_r1.llm_agent.sexpr_config import SExprGenerationConfig
# Import S-Expression generation manager
from kbqa_r1.llm_agent.sexpr_generation import SExprLLMGenerationManager
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score import mid_reward
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _select_rm_score_fn(data_source):
    """Select reward model score function based on data source."""
    if data_source in ['webqsp', 'grailqa', 'graphq']:
        return mid_reward.compute_mid_reward
    else:
        raise NotImplementedError(f"Unsupported data source for KBQA: {data_source}")


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation_kbqa(config)


def run_generation_kbqa(config) -> None:
    """Main entry point for KBQA generation with reward computation."""
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "SEXPR_MODE": "true",
                "ENABLE_ACTION_REASONING": "true",
            }
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task_kbqa.remote(config))


@ray.remote(num_cpus=1)
def main_task_kbqa(config):
    """Main task for KBQA generation with S-Expression and reward computation."""
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Load tokenizer
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # Validate configuration
    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # Load dataset
    dataset = pd.read_parquet(config.data.path)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Extract necessary columns
    prompts = dataset[config.data.prompt_key].tolist()
    data_source = config.data.get("data_source", "webqsp")
    
    # Get ground truth if available
    if "ground_truth" in dataset.columns:
        ground_truths = dataset["ground_truth"].tolist()
    else:
        ground_truths = [[] for _ in range(len(dataset))]
        print("Warning: No ground_truth column found, rewards will be 0")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize worker group with rollout-only mode
    # For generation-only mode, use smaller max_colocate_count to reduce CPU requirements
    # max_colocate_count controls CPU allocation per GPU worker
    # Generation mode needs fewer CPUs than training mode (推理比训练需要更少的 CPU)
    max_colocate_count = config.get("max_colocate_count", 8)  # Default 4 CPUs per GPU for generation
    
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config,
        role="rollout"
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        max_colocate_count=max_colocate_count  # Explicitly set CPU allocation
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()
    
    # Print resource allocation for debugging
    total_cpus_required = config.trainer.n_gpus_per_node * max_colocate_count
    print(f"[RESOURCE] GPUs: {config.trainer.n_gpus_per_node}, "
          f"CPUs per GPU: {max_colocate_count}, "
          f"Total CPUs required: {total_cpus_required}")
    
    # Initialize S-Expression generation manager
    sexpr_config = SExprGenerationConfig(
        enable_sexpr_mode=True,
        enable_action_reasoning=True,
        enable_relation_retrieval=True,
        max_turns=config.rollout.get("max_turns", 10),
        max_prompt_length=config.rollout.prompt_length,
        max_response_length=config.rollout.response_length,
    )
    
    sexpr_manager = SExprLLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=wg,
        config=sexpr_config,
        is_validation=True,  # Generation mode, no training
        dataset=data_source,
    )
    print(f"Initialized SExprLLMGenerationManager for {data_source}")

    # Batch processing
    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    
    # Results storage: [n_samples, n_data]
    output_responses = [[] for _ in range(config.data.n_samples)]
    output_rewards = [[] for _ in range(config.data.n_samples)]
    output_reward_details = [[] for _ in range(config.data.n_samples)]

    # Select reward function
    compute_score_fn = _select_rm_score_fn(data_source)
    print(f"Using reward function for data_source: {data_source}")

    for batch_idx in tqdm(range(num_batch), desc="Processing batches"):
        print(f"\n[{batch_idx + 1}/{num_batch}] Processing batch...")
        
        batch_start = batch_idx * config_batch_size
        batch_end = min((batch_idx + 1) * config_batch_size, total_samples)
        batch_prompts = prompts[batch_start:batch_end]
        batch_ground_truths = ground_truths[batch_start:batch_end]

        # Tokenize prompts
        if isinstance(batch_prompts[0], str):
            # Simple string prompts
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
            )
        else:
            # Chat format prompts
            apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
            inputs = tokenizer.apply_chat_template(
                batch_prompts,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # Generate n_samples responses for each prompt using run_llm_loop
        print(f"[{batch_idx + 1}/{num_batch}] Generating {config.data.n_samples} samples...")
        for n_sample in range(config.data.n_samples):
            # Use SExprLLMGenerationManager's run_llm_loop
            output_padded = sexpr_manager.run_llm_loop(data_padded, input_ids)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            # Extract responses and compute rewards
            batch_responses = []
            batch_batch_rewards = []
            batch_reward_details = []
            
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                # Compute reward
                ground_truth = batch_ground_truths[i]
                try:
                    reward_dict = compute_score_fn(
                        response_str,
                        ground_truth,
                        format_score=config.get("format_score", 0.0),
                        structure_format_score=config.get("structure_format_score", 0.1),
                    )
                    
                    if isinstance(reward_dict, dict):
                        total_reward = reward_dict.get("total", 0.0)
                        reward_details = reward_dict
                    else:
                        total_reward = float(reward_dict)
                        reward_details = {"total": total_reward}
                        
                except Exception as e:
                    print(f"Warning: Failed to compute reward for sample {i}: {e}")
                    total_reward = 0.0
                    reward_details = {"total": 0.0, "error": str(e)}

                batch_responses.append(response_str)
                batch_batch_rewards.append(total_reward)
                batch_reward_details.append(reward_details)

            output_responses[n_sample].extend(batch_responses)
            output_rewards[n_sample].extend(batch_batch_rewards)
            output_reward_details[n_sample].extend(batch_reward_details)
            
            # Print sample statistics for this n_sample
            if batch_batch_rewards:
                print(f"  Sample {n_sample+1}/{config.data.n_samples}: "
                      f"mean_reward={np.mean(batch_batch_rewards):.3f}, "
                      f"max={np.max(batch_batch_rewards):.3f}")

    # Convert from [n_samples, n_data] to [n_data, n_samples]
    output_responses = np.array(output_responses, dtype=object).T.tolist()
    output_rewards = np.array(output_rewards, dtype=object).T.tolist()
    output_reward_details = np.array(output_reward_details, dtype=object).T.tolist()

    # Add to dataframe
    dataset["responses"] = output_responses
    dataset["rewards"] = output_rewards
    dataset["reward_details"] = output_reward_details

    # Select best response for each sample based on reward
    best_responses = []
    best_rewards = []
    best_reward_details = []
    
    for i in range(len(dataset)):
        sample_rewards = output_rewards[i]
        if sample_rewards:
            best_idx = np.argmax(sample_rewards)
            best_responses.append(output_responses[i][best_idx])
            best_rewards.append(sample_rewards[best_idx])
            best_reward_details.append(output_reward_details[i][best_idx])
        else:
            best_responses.append("")
            best_rewards.append(0.0)
            best_reward_details.append({})
    
    dataset["best_response"] = best_responses
    dataset["best_reward"] = best_rewards
    dataset["best_reward_details"] = best_reward_details

    # Print statistics
    print("\n" + "="*60)
    print("GENERATION STATISTICS")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    print(f"Samples per prompt: {config.data.n_samples}")
    
    if best_rewards:
        print(f"\nBest Reward Statistics:")
        print(f"  Mean: {np.mean(best_rewards):.4f}")
        print(f"  Std:  {np.std(best_rewards):.4f}")
        print(f"  Max:  {np.max(best_rewards):.4f}")
        print(f"  Min:  {np.min(best_rewards):.4f}")
        
        # Count high-quality samples
        reward_threshold = config.get("reward_threshold", 0.8)
        high_quality_count = sum(1 for r in best_rewards if r >= reward_threshold)
        print(f"\nHigh-quality samples (reward >= {reward_threshold}): "
              f"{high_quality_count} / {len(dataset)} "
              f"({100*high_quality_count/len(dataset):.1f}%)")

    # Save outputs
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    
    # Save full dataset with all samples
    dataset.to_parquet(config.data.output_path)
    print(f"\nSaved full results to: {config.data.output_path}")
    
    # Optionally save high-quality samples only
    if config.get("save_high_quality_only", False):
        reward_threshold = config.get("reward_threshold", 0.8)
        high_quality_dataset = dataset[dataset["best_reward"] >= reward_threshold].copy()
        
        if len(high_quality_dataset) > 0:
            hq_output_path = config.data.output_path.replace(".parquet", "_high_quality.parquet")
            high_quality_dataset.to_parquet(hq_output_path)
            print(f"Saved high-quality samples to: {hq_output_path}")
            print(f"  ({len(high_quality_dataset)} samples)")
        else:
            print(f"Warning: No high-quality samples found (threshold={reward_threshold})")
    
    # Save summary statistics
    summary_path = config.data.output_path.replace(".parquet", "_summary.json")
    summary = {
        "total_samples": len(dataset),
        "n_samples_per_prompt": config.data.n_samples,
        "data_source": data_source,
        "reward_statistics": {
            "mean": float(np.mean(best_rewards)) if best_rewards else 0.0,
            "std": float(np.std(best_rewards)) if best_rewards else 0.0,
            "max": float(np.max(best_rewards)) if best_rewards else 0.0,
            "min": float(np.min(best_rewards)) if best_rewards else 0.0,
        },
        "high_quality_count": sum(1 for r in best_rewards if r >= config.get("reward_threshold", 0.8)),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    
    print("="*60)
    print("GENERATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
