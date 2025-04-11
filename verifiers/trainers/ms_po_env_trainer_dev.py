from typing import Callable, Optional, Union, Any, List, Dict, Tuple
import logging

import torch
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from trl import GRPOConfig

from verifiers.envs.environment import Environment
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer, RewardFunc

if is_peft_available():
    from peft import PeftConfig  # type: ignore

if is_wandb_available():
    import wandb

class MSPOEnvTrainer_Dev(GRPOEnvTrainer):
    """
    Multi-Step PO Environment Trainer that calculates separate advantages for 
    step rewards and outcome rewards. If result tag exists:
    - Tokens before a '<result>' tag get total advantage (step+outcome)
    - Tokens after only get outcome advantage
    If no result tag exists:
    - All tokens get total advantage
    """
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            step_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            outcome_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            step_reward_weights: Optional[List[float]] = None,
            outcome_reward_weights: Optional[List[float]] = None,
            discount_factor: float = 1.0,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        # Convert single reward functions to lists
        if callable(step_reward_funcs):
            step_reward_funcs = [step_reward_funcs]
        if callable(outcome_reward_funcs):
            outcome_reward_funcs = [outcome_reward_funcs]
            
        # Create combined reward funcs for parent class
        self.step_reward_funcs = step_reward_funcs
        self.outcome_reward_funcs = outcome_reward_funcs
        self.discount_factor = discount_factor
        combined_reward_funcs = step_reward_funcs + outcome_reward_funcs
        
        # Set up reward weights
        self.num_step_funcs = len(step_reward_funcs)
        self.num_outcome_funcs = len(outcome_reward_funcs)
        
        # Set reward weights or use defaults (all ones)
        self.step_reward_weights = torch.ones(self.num_step_funcs)
        self.outcome_reward_weights = torch.ones(self.num_outcome_funcs)
        
        super().__init__(
            model=model,
            env=env,
            reward_funcs=combined_reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

    def _generate_and_score_completions(
         self, inputs: Dict[str, Union[torch.Tensor, Any]]   
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]  # type: ignore 
        
        # Generate completions using the environment
        prompt_inputs, prompt_ids, prompt_mask = self._prepare_prompt_inputs(inputs)
        
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate completions using the environment
        all_prompts, completion_ids, completion_messages, completion_mask = self._generate_completions(prompts)
        
        # print one completion with full tensor
        #print('completion_messages[0]', completion_messages[0])
        #print('completion_ids[0]', completion_ids[0])
        
        # let's verify
        # for one completion_mask, it should be like [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]
        # in this case, we see that such completion mask has 4 segments
        # let's check whether the number of messages in one completion_messages is equal to the number of segments - 1 in completion_mask
        # for example, the completion_messages is like [..., {'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, {'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...],
        
        '''
        import torch
        torch.set_printoptions(threshold=float('inf'))  # 禁用省略
        
        def verify_completion_mask_and_messages(completion_mask, completion_messages):
            # Count segments in completion_mask
            segments = 1  # Start with 1 for the first segment
            segment_boundaries = [0]  # Record where segments change
            
            for i in range(1, len(completion_mask)):
                if completion_mask[i] != completion_mask[i-1]:
                    segments += 1
                    segment_boundaries.append(i)
            
            # Add final boundary
            segment_boundaries.append(len(completion_mask))
            
            # Count messages in completion_messages
            message_count = len(completion_messages)
            
            # Extract segment values and lengths
            segment_values = []
            segment_lengths = []
            
            for i in range(len(segment_boundaries)-1):
                start = segment_boundaries[i]
                end = segment_boundaries[i+1]
                value = completion_mask[start].item()
                length = end - start
                segment_values.append(value)
                segment_lengths.append(length)
            
            # Prepare debug information
            print("\n" + "="*80)
            print(f"MASK ANALYSIS:")
            print(f"Total mask length: {len(completion_mask)}")
            print(f"Number of segments: {segments}")
            print(f"Number of messages: {message_count}")
            print(f"Messages roles: {[m.get('role', 'unknown') for m in completion_messages]}")
            print(f"Segment boundaries: {segment_boundaries}")
            print(f"Segment values: {segment_values}")
            print(f"Segment lengths: {segment_lengths}")
            
            # Print mask pattern visualization
            pattern = ''.join(str(x) for x in segment_values)
            print(f"Mask pattern: {pattern}")
            
            # Print message-segment alignment
            print("\nMessage to Segment Alignment:")
            print("Expected alignment: Each message corresponds to a boundary between segments")
            
            # 修改验证条件，允许两种情况：消息数 = 段数-1 或 消息数 = 段数
            condition1 = message_count == segments - 1
            condition2 = message_count == segments
            
            print(f"Case 1: messages count ({message_count}) == segments count - 1 ({segments - 1}): {condition1}")
            print(f"Case 2: messages count ({message_count}) == segments count ({segments}): {condition2}")
            
            # Print message contents briefly
            print("\nFirst 50 chars of each message:")
            for i, msg in enumerate(completion_messages):
                content = msg.get('content', '')
                print(f"  Message {i} ({msg.get('role', 'unknown')}): {content[:50]}...")
            
            # Print the first few and last few tokens of completion_mask
            print("\nMask samples:")
            print(f"First 20: {completion_mask[:20].tolist()}")
            print(f"Last 20: {completion_mask[-20:].tolist()}")
            
            # 验证条件更新：messages数量等于segments数量减1或messages数量等于segments数量
            if not (condition1 or condition2):
                print(f"\nVERIFICATION FAILED: Number of messages ({message_count}) does not match either")
                print(f"segments - 1 ({segments - 1}) or segments ({segments})")
                # Continue execution (return True) instead of raising error
                return True
            
            print("\nVERIFICATION PASSED!")
            return True
        
        # Apply verification to each item in batch
        for i in range(len(completion_mask)):
            verify_completion_mask_and_messages(completion_mask[i], completion_messages[i])
            # Only check the first item
            if i == 0:
                break
        
        # Continue execution instead of quitting
        quit()
        '''
        
        # Prepare model inputs
        prompt_completion_ids, attention_mask, logits_to_keep = self._prepare_model_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        
        # Compute logps
        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_completion_ids, attention_mask, logits_to_keep
        )
        
        # Calculate step rewards and outcome rewards separately 
        rewards_step = self._calculate_rewards(
            prompts, completion_messages, self.step_reward_funcs, inputs
        )
        
        rewards_outcome = self._calculate_rewards(
            prompts, completion_messages, self.outcome_reward_funcs, inputs
        )
        
        # Apply weights and sum
        step_rewards = (rewards_step * self.step_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        outcome_rewards = (rewards_outcome * self.outcome_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        # Calculate total rewards (step + outcome) 
        total_rewards = step_rewards + self.discount_factor * outcome_rewards
        
        # Compute normalized advantages
        total_advantages = self._compute_normalized_advantages(total_rewards, len(prompts))
        outcome_advantages = self._compute_normalized_advantages(outcome_rewards, len(prompts))
        
        # Find the segment indices that contain '<result>' tags in each completion
        result_segment_indices = self._find_result_positions(completion_ids, completion_messages)
        
        # Apply the advantages based on result segment indices
        # If there's a result tag, tokens before get total advantage, after get outcome 
        # If no result tag, all tokens get total advantage 
        combined_advantages = self._combine_advantages(
            completion_mask, total_advantages, outcome_advantages, result_segment_indices
        )
        
        # Log metrics
        self._log_metrics(
            prompts, completion_messages, completion_mask, 
            rewards_step, rewards_outcome, step_rewards, outcome_rewards, total_rewards
        )
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": combined_advantages,
        }
    
    def _prepare_prompt_inputs(self, inputs):
        """Prepare the prompt inputs for the model."""
        from trl.data_utils import maybe_apply_chat_template
        from transformers import Trainer
        
        prompts = [x["prompt"] for x in inputs]  # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]  # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            
        return prompt_inputs, prompt_ids, prompt_mask
    
    def _generate_completions(self, prompts):
        """Generate completions using the environment and broadcast the results."""
        from accelerate.utils import broadcast_object_list, gather_object
        
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        
        # Convert to tensors and pad
        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = self._pad(completion_ids, padding_value=self.processing_class.pad_token_id)

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = self._pad(completion_mask, padding_value=0)
        
        return all_prompts, completion_ids, completion_messages, completion_mask
    
    def _pad(self, tensors, padding_value=0):
        """Pad a list of tensors to the same length."""
        from trl.trainer.utils import pad
        return pad(tensors, padding_value=padding_value)
    
    def _prepare_model_inputs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """Prepare the model inputs for logit computation."""
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        return prompt_completion_ids, attention_mask, logits_to_keep
    
    def _compute_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        """Compute log probabilities using the model and reference model."""
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                    
        return old_per_token_logps, ref_per_token_logps
    
    def _calculate_rewards(self, prompts, completions, reward_funcs, inputs):
        """Calculate rewards for a set of reward functions."""
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device)
        
        for i, reward_func in enumerate(reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]  # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}  # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)  # type: ignore
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            
        from accelerate.utils import gather
        return gather(rewards_per_func)
    
    def _compute_normalized_advantages(self, rewards, slice_length=None):
        """Compute normalized advantages from rewards."""
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * slice_length,
            (self.accelerator.process_index + 1) * slice_length,
        )
        return advantages[process_slice]
    
    def _find_result_positions(self, completion_ids, completion_messages):
        """
        Find the segment index that contains '<result>' tags in completions.
        
        Instead of returning token positions, now returns segment indices.
        - If a user message contains '<result>', return the index of that user message
        - If no result tag is found, return -1
        """
        result_segment_indices = []
        
        for i, messages in enumerate(completion_messages):
            result_segment = -1
            
            # Handle dialogue history format
            if isinstance(messages, list):
                # Look for assistant message followed by user message (env response)
                for j, msg in enumerate(messages):
                    if msg.get('role') == 'assistant':
                        # Check if there's a subsequent environment response (user message)
                        if j + 1 < len(messages) and messages[j + 1].get('role') == 'user':
                            user_msg = messages[j + 1].get('content', '')
                            
                            # Check if user message contains a '<result>' tag
                            if '<result>' in user_msg:
                                # Store the segment index of this user message
                                result_segment = j + 1
                                break
            
            # Handle string format (kept for compatibility)
            elif isinstance(messages, str):
                # Raise error for unsupported format
                raise ValueError("Completion is a string, which is not supported.")
            
            result_segment_indices.append(result_segment)
            
        return result_segment_indices
    
    def _combine_advantages(self, completion_mask, total_advantages, outcome_advantages, result_segment_indices):
        """
        Combine total and outcome advantages based on result segment indices.
        
        For each trajectory:
        - If result_segment > 0: all segments before result_segment get total advantage, after get outcome advantage
        - If result_segment = -1: all segments get total advantage
        
        Note: We don't multiply by completion_mask here as it will be applied in compute_loss
        """
        device = self.accelerator.device
        batch_size, seq_len = completion_mask.shape
        combined_advantages = torch.zeros_like(completion_mask, dtype=torch.float32)
        
        for i in range(batch_size):
            result_segment = result_segment_indices[i]
            
            # Expand scalar advantages to sequence length
            total_advantage_expanded = total_advantages[i].item() * torch.ones_like(completion_mask[i], dtype=torch.float32)
            outcome_advantage_expanded = outcome_advantages[i].item() * torch.ones_like(completion_mask[i], dtype=torch.float32)
            
            if result_segment > 0:
                # 查找所有段落的边界
                segment_boundaries = [0]  # 第一个段落的起始位置
                current_segment = 0
                
                # 遍历mask找出所有段落边界
                for j in range(1, seq_len):
                    if completion_mask[i][j] != completion_mask[i][j-1]:
                        current_segment += 1
                        segment_boundaries.append(j)
                        
                # 添加序列结束位置作为最后一个边界
                segment_boundaries.append(seq_len)
                
                # 如果找到了足够多的段落边界
                if result_segment < len(segment_boundaries):
                    # 获取result_segment对应的起始位置
                    split_point = segment_boundaries[result_segment]
                    
                    # 创建分割掩码
                    before_result_mask = torch.zeros(seq_len, device=device)
                    before_result_mask[:split_point] = 1.0
                    
                    # 后面部分的掩码是补集
                    after_result_mask = 1.0 - before_result_mask
                    
                    combined_advantages[i] = (total_advantage_expanded * before_result_mask) + (outcome_advantage_expanded * after_result_mask)
                else:
                    # we can raise an error here
                    raise ValueError(f"No enough segments found in completion {i}")

            else:
                # 没有result标签，所有token都使用total_advantage
                combined_advantages[i] = total_advantage_expanded
                
        return combined_advantages
    
    def _log_metrics(self, prompts, completions, completion_mask, 
                    rewards_step, rewards_outcome, step_rewards, outcome_rewards, total_rewards):
        """Log metrics for step, outcome, and total rewards."""
        mode = "eval" if self.control.should_evaluate else "train"

        # Log completion length
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Log individual reward functions
        for i, reward_func in enumerate(self.step_reward_funcs):
            reward_func_name = getattr(reward_func, "__name__", f"step_reward_{i}")
            self._metrics[mode][f"rewards/step/{reward_func_name}"].append(rewards_step.mean(0)[i].item())
            
        for i, reward_func in enumerate(self.outcome_reward_funcs):
            reward_func_name = getattr(reward_func, "__name__", f"outcome_reward_{i}")
            self._metrics[mode][f"rewards/outcome/{reward_func_name}"].append(rewards_outcome.mean(0)[i].item())

        # Log overall rewards
        self._metrics[mode]["reward/step"].append(step_rewards.mean().item())
        self._metrics[mode]["reward/outcome"].append(outcome_rewards.mean().item())
        self._metrics[mode]["reward/total"].append(total_rewards.mean().item())
        
        # Log samples if needed
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completion_samples(prompts, completions, total_rewards)
            
    def _log_completion_samples(self, prompts, completions, rewards):
        """Log completion samples to console and wandb if available."""
        from accelerate.utils import gather_object
        
        prompts_to_log = gather_object(prompts)
        completions_to_log = gather_object(completions)
        rewards_to_log = rewards.tolist()

        if self.accelerator.is_main_process:
            if len(prompts_to_log) > 0:
                from trl.import_utils import is_rich_available
                
                if is_rich_available():
                    from verifiers.utils.logging_utils import print_prompt_completions_sample
                    
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                    
                if self.args.report_to and "wandb" in self.args.report_to and is_wandb_available() and wandb.run is not None:
                    import pandas as pd
                    
                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The MSPOEnvTrainer does not support returning outputs")
            
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        
        # Check advantages dimensions - if 2D, it's already expanded
        if len(advantages.shape) == 2:
            # Already expanded advantages, use directly
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
        else:
            # Unexpanded advantages, expand with unsqueeze(1)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        # Add detailed shape logging
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            logging.info(f"Step: {self.state.global_step}")
            logging.info(f"advantages shape: {advantages.shape}")
            logging.info(f"old_per_token_logps shape: {old_per_token_logps.shape}")
            logging.info(f"per_token_logps shape: {per_token_logps.shape}")
            logging.info(f"coef_1 shape: {coef_1.shape}")
            logging.info(f"coef_2 shape: {coef_2.shape}")
            logging.info(f"per_token_loss1 shape: {per_token_loss1.shape}")
            logging.info(f"per_token_loss2 shape: {per_token_loss2.shape}")
            logging.info(f"per_token_loss shape: {per_token_loss.shape}")
            if self.beta != 0.0:
                logging.info(f"per_token_loss after KL shape: {per_token_loss.shape}")
            logging.info(f"final loss shape: {loss.shape}")

        return loss