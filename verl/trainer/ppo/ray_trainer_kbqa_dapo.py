"""KBQA R1 DAPO-style filtered trainer (minimal changes).

This subclass adds prompt-level multi-generation filtering (std>0 rule) to the
original KBQA RayPPOTrainer without touching its source file. Only the
generation + early reward region is wrapped in a while loop when
algorithm.filter_groups.enable is True.
"""

import re  # for state masking regex (if used by base class helpers)
import uuid
from collections import defaultdict
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           reduce_metrics)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.ray_trainer_kbqa import (RayPPOTrainer, apply_kl_penalty,
                                               compute_advantage,
                                               compute_response_mask,
                                               should_save_ckpt_esi)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import Tracking

try:
    from kbqa_r1.llm_agent.sexpr_generation import (SExprGenerationConfig,
                                                    SExprLLMGenerationManager)
    _HAS_SEXPR = True
except Exception:
    _HAS_SEXPR = False


class RayPPOTrainerKBQADAPO(RayPPOTrainer):
    def fit(self):  # noqa: C901 retain structure
        from pprint import pprint

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            RolloutSkip(self.config, self.actor_rollout_wg).wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        filter_cfg = getattr(self.config.algorithm, "filter_groups", None)
        filter_enabled = bool(filter_cfg and getattr(filter_cfg, "enable", False))

        # If filtering not enabled, fall back to original trainer behavior.
        if not filter_enabled:
            return super().fit()

        # ================= Filter-enabled training loop (DAPO style) =================
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress (filter)")
        # we start from step 1 (already incremented once above)
        last_val_metrics = None
        prev_step_profile = False
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = defaultdict(float)

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # Immutable prompt batch for successive generation attempts
                prompt_batch: DataProto = DataProto.from_single_dict(batch_dict)
                if "uid" not in prompt_batch.non_tensor_batch:
                    prompt_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(prompt_batch.batch))], dtype=object
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                num_prompt_in_batch = 0
                num_gen_batches = 0
                accumulated_batch = None
                reward_extra_infos_dict = {}

                with marked_timer("step", timing_raw):
                    while True:  # multi-generation filtering loop
                        num_gen_batches += 1
                        gen_batch = self._get_gen_batch(prompt_batch)
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        gen_batch = gen_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )

                        enable_do_search = self.config.get("do_search", False)
                        if not enable_do_search:
                            with marked_timer("gen", timing_raw, color="red"):
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                else:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                                gen_batch_output.meta_info.pop("timing", None)
                        else:
                            enable_sexpr_mode = self.config.get("sexpr_config", {}).get("enable_sexpr_mode", False)
                            if enable_sexpr_mode and _HAS_SEXPR:
                                max_start_length = self.config.data.get(
                                    "max_start_length", self.config.data.max_prompt_length // 2
                                )
                                max_obs_length = self.config.data.get("max_obs_length", 512)
                                gen_config = SExprGenerationConfig(
                                    max_turns=self.config.get("max_turns", 10),
                                    max_start_length=max_start_length,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    max_response_length=self.config.data.max_response_length,
                                    max_obs_length=max_obs_length,
                                    num_gpus=self.config.trainer.n_gpus_per_node,
                                    no_think_rl=self.config.algorithm.get("no_think_rl", False),
                                    enable_sexpr_mode=True,
                                    enable_action_validation=self.config.get("sexpr_config", {}).get(
                                        "enable_action_reasoning", True
                                    ),
                                    enable_sexpr_validation=self.config.get("sexpr_config", {}).get(
                                        "enable_semantic_validation", True
                                    ),
                                    hyper_r1_enable=self.config.get("hyper_r1", {}).get("enable", False),
                                    hyper_r1_max_active=self.config.get("hyper_r1", {}).get("max_active", 6),
                                    hyper_r1_max_nodes=self.config.get("hyper_r1", {}).get("max_nodes", 24),
                                    hyper_r1_frontier_width=self.config.get("hyper_r1", {}).get("frontier_width", 3),
                                    hyper_r1_max_frontier_width=self.config.get("hyper_r1", {}).get("max_frontier_width", 6),
                                    hyper_r1_relation_model=self.config.get("hyper_r1", {}).get("relation_model"),
                                    sparql_url=self.config.get("sparql", {}).get(
                                        "url", "http://localhost:8000/execute"
                                    ),
                                    use_odbc=self.config.get("use_odbc", False),
                                    use_aioodbc=self.config.get("use_aioodbc", True),
                                    odbc_config=self.config.get("odbc_config", None),
                                    experiment_name=self.config.trainer.experiment_name,
                                    current_step=self.global_steps,
                                )
                                generation_manager = SExprLLMGenerationManager(
                                    tokenizer=self.tokenizer,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    config=gen_config,
                                    sparql_config={
                                        "sparql_url": self.config.get("sparql", {}).get(
                                            "url", "http://localhost:8000/execute"
                                        ),
                                        "sparql_batch_size": self.config.get("sparql_batch_size", 128),
                                        "sparql_max_concurrent": self.config.get("sparql_max_concurrent", 16),
                                    },
                                )
                                first_input_ids = gen_batch.batch["input_ids"][
                                    :, -gen_config.max_start_length:
                                ].clone().long()
                                with marked_timer("gen", timing_raw, color="red"):
                                    generation_manager.config.current_step = self.global_steps
                                    generation_manager.timing_raw = timing_raw
                                    gen_batch_output = generation_manager.run_llm_loop(
                                        gen_batch=gen_batch, initial_input_ids=first_input_ids
                                    )
                                for key in gen_batch_output.batch.keys():
                                    gen_batch_output.batch[key] = gen_batch_output.batch[key].long()
                                with torch.no_grad():
                                    lp_out = self.actor_rollout_wg.compute_log_prob(gen_batch_output)
                                    gen_batch_output = gen_batch_output.union(lp_out)
                            else:
                                with marked_timer("gen", timing_raw, color="red"):
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                    timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                                    gen_batch_output.meta_info.pop("timing", None)

                        # REMAX baseline
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("REMAX requires reward_fn.")
                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = (
                                    self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                    if not self.async_rollout_mode
                                    else self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                )
                                baseline_container = prompt_batch.repeat(
                                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                                ).union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(baseline_container).sum(dim=-1)
                                del gen_baseline_batch, gen_baseline_output

                        attempt_batch = prompt_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        ).union(gen_batch_output)
                        if (
                            self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX
                            and "reward_baseline_tensor" in locals()
                        ):
                            attempt_batch.batch["reward_baselines"] = reward_baseline_tensor

                        # Early reward (for filtering)
                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm and "rm_scores" not in attempt_batch.batch:
                                rm_scores = self.rm_wg.compute_rm_score(attempt_batch)
                                attempt_batch = attempt_batch.union(rm_scores)
                            if self.config.reward_model.launch_reward_fn_async:
                                fut = compute_reward_async.remote(data=attempt_batch, reward_fn=self.reward_fn)
                                reward_tensor, reward_extra_infos_dict = ray.get(fut)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(attempt_batch, self.reward_fn)
                            attempt_batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                attempt_batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )
                        if "response_mask" not in attempt_batch.batch:
                            attempt_batch.batch["response_mask"] = compute_response_mask(attempt_batch)

                        # Filtering logic
                        metric_name = getattr(filter_cfg, "metric", "seq_reward")
                        if metric_name == "seq_final_reward" and self.config.algorithm.use_kl_in_reward:
                            metric_name = "seq_reward"  # cannot compute final reward yet
                        if metric_name == "seq_final_reward" and "token_level_rewards" in attempt_batch.batch:
                            attempt_batch.non_tensor_batch["seq_final_reward"] = (
                                attempt_batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                            )
                        else:
                            attempt_batch.non_tensor_batch["seq_reward"] = (
                                attempt_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                            )
                        use_key = metric_name if metric_name == "seq_final_reward" else "seq_reward"
                        uid2vals = defaultdict(list)
                        for uid, val in zip(
                            attempt_batch.non_tensor_batch["uid"],
                            attempt_batch.non_tensor_batch[use_key],
                            strict=True,
                        ):
                            uid2vals[uid].append(val)
                        uid2std = {u: np.std(v) for u, v in uid2vals.items()}
                        kept_uids = [u for u, s in uid2std.items() if s > 0 or len(uid2vals[u]) == 1]
                        kept_indices = [
                            i for i, uid in enumerate(attempt_batch.non_tensor_batch["uid"]) if uid in kept_uids
                        ]
                        attempt_batch = attempt_batch[kept_indices]
                        num_prompt_in_batch += len(kept_uids)
                        accumulated_batch = (
                            attempt_batch
                            if accumulated_batch is None
                            else DataProto.concat([accumulated_batch, attempt_batch])
                        )

                        if num_prompt_in_batch < self.config.data.train_batch_size:
                            max_num = getattr(filter_cfg, "max_num_gen_batches", 0)
                            if max_num <= 0 or num_gen_batches < max_num:
                                continue  # generate again
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num=}. Insufficient prompts kept after filtering."
                                )
                        else:
                            traj_bsz = (
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n
                            )
                            batch = accumulated_batch[:traj_bsz]
                            break  # exit filtering loop

                # ===== Post-generation (final batch ready) =====
                if "response_mask" not in batch.batch:
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1
                ).tolist()

                # Old log prob & entropy
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_lp = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_lp.batch["entropys"]
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=batch.batch["response_mask"],
                        loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                    )
                    metrics.update({"actor/entropy": float(entropy_agg.detach().item())})
                    old_lp.batch.pop("entropys")
                    batch = batch.union(old_lp)

                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        ref_lp = (
                            self.ref_policy_wg.compute_ref_log_prob(batch)
                            if not self.ref_in_actor
                            else self.actor_rollout_wg.compute_ref_log_prob(batch)
                        )
                        batch = batch.union(ref_lp)

                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    if self.config.algorithm.use_kl_in_reward and "token_level_rewards" not in batch.batch:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    elif "token_level_rewards" not in batch.batch:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # DEBUG: Log before computing rollout importance weights
                    logger.info(f"[MISMATCH DEBUG] Checking rollout IS configuration:")
                    logger.info(f"  rollout_is_threshold = {self.config.algorithm.rollout_is_threshold}")
                    logger.info(f"  rollout_is = {self.config.algorithm.get('rollout_is', False)}")
                    logger.info(f"  'rollout_log_probs' in batch = {'rollout_log_probs' in batch.batch}")
                    logger.info(f"  'old_log_probs' in batch = {'old_log_probs' in batch.batch}")
                    if 'rollout_log_probs' in batch.batch:
                        logger.info(f"  rollout_log_probs shape = {batch.batch['rollout_log_probs'].shape}")
                    if 'old_log_probs' in batch.batch:
                        logger.info(f"  old_log_probs shape = {batch.batch['old_log_probs'].shape}")

                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    
                    # DEBUG: Log after computing rollout importance weights
                    logger.info(f"[MISMATCH DEBUG] After compute_rollout_importance_weights:")
                    logger.info(f"  is_metrics keys = {list(is_metrics.keys())}")
                    if is_metrics:
                        for k, v in is_metrics.items():
                            logger.info(f"    {k} = {v}")
                    else:
                        logger.warning(f"  ❌ is_metrics is EMPTY! Mismatch metrics were not computed!")
                    
                    metrics.update(is_metrics)

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=self.config.algorithm,
                    )

                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        if self.config.get("do_search", False) and self.config.actor_rollout_ref.actor.get(
                            "state_masking", False
                        ):
                            batch, metrics = self._create_loss_mask(batch, metrics)
                        batch.meta_info["multi_turn"] = (
                            self.config.actor_rollout_ref.rollout.multi_turn.enable
                        )
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Checkpoint save
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI expiration approaching.")
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

                steps_duration = timing_raw["step"] if "step" in timing_raw else 0.0
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Aggregate metrics
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()
                    )
                )
                metrics["train/num_gen_batches"] = num_gen_batches

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

        progress_bar.close()
