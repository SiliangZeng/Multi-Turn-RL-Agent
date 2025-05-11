import argparse
import torch
import verifiers as vf
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from verifiers.tools import wiki_search
from verifiers.prompts import DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import TrivialQAToolRubric

from vllm import LLM, SamplingParams
from tqdm import tqdm


def calculate_rewards(prompts, completions, reward_funcs, inputs):
    rewards_per_func = torch.zeros(len(prompts), len(reward_funcs))

    for i, reward_func in enumerate(reward_funcs):
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        output_reward_func = reward_func(
            prompts=prompts, completions=completions, **reward_kwargs
        )
        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32)

    return rewards_per_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the LLM with tools")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Model name or path (default: Qwen/Qwen2.5-7B)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default: 8)",
    )

    args = parser.parse_args()

    vf_env = vf.ToolEnv(
        dataset="triviaqa",
        tools=[wiki_search],
        system_prompt=DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE,
        max_steps=2,
    )

    eval_dataset = vf_env.get_eval_dataset()
    print(f"dataset size: {len(eval_dataset)}")

    rubric_class = TrivialQAToolRubric()

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.7,
        dtype="auto",
        enable_prefix_caching=True,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        max_tokens=1024,
        guided_decoding=None,
        n=1,
        temperature=0.9,
        top_p=1,
        top_k=50,
        min_p=0,
        repetition_penalty=1,
    )

    def data_collator(features):
        return features

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
    )

    all_rewards = []
    for batch in tqdm(dataloader, desc=f"Inference"):
        prompts = [x["prompt"] for x in batch]
        env_result = vf_env.generate_only_msg(
            prompts=prompts, llm=llm, sampling_params=sampling_params
        )
        completion_messages = env_result["messages"]
        
        print("One completion message:", env_result["messages"][0])
        rewards = calculate_rewards(
            prompts=prompts,
            completions=completion_messages,
            reward_funcs=rubric_class.get_reward_funcs(),
            inputs=batch,
        )
        all_rewards.append(rewards)
        print("Rewards:", rewards)

    final_rewards = torch.cat(all_rewards, dim=0)
    
    print("Final Rewards shape:", final_rewards.shape)
    print("Final Rewards:", final_rewards)
    
    mean_rewards = final_rewards.mean(dim=0)
    print("Mean Rewards shape:", mean_rewards.shape)
    print("Mean Rewards:", mean_rewards)
