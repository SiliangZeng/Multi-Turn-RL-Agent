import argparse
import verifiers as vf

from verifiers.tools import wiki_search
from verifiers.prompts import DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import TrivialQAToolRubric


parser = argparse.ArgumentParser(
    description="Evaluate a model on the TriviaQA dataset with wiki search tool."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-7B",
    help="Model name or path (default: Qwen/Qwen2.5-7B)",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=64,
    help="Batch size for evaluation (default: 1)",
)
parser.add_argument(
    "--subfolder",
    type=str,
    default="checkpoint-300",
    help="Subfolder of the model to use (default: checkpoint-300)",
)
args = parser.parse_args()

model_name = args.model_name
model, tokenizer = vf.get_model_and_tokenizer(model_name, subfolder=subfolder)

vf_env = vf.ToolEnv(
    dataset="triviaqa",
    tools=[wiki_search],
    system_prompt=DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE,
    max_steps=2,
)

eval_dataset = vf_env.get_eval_dataset()
print(f"Eval dataset size: {len(eval_dataset)}")

rubric_class = TrivialQAToolRubric()

turn_reward_funcs = [
    rubric_class.tool_execution_reward_func,
    rubric_class.exist_answer_in_search_results,
]

outcome_reward_funcs = [
    rubric_class.exist_answer_reward_func,
    rubric_class.exact_match_reward_func,
    rubric_class.parser.get_format_reward_func(),
    rubric_class.parser.get_xml_reward_func(),
]


training_args = vf.get_default_grpo_config(
    num_gpus=2,
)
training_args.report_to = "none"
training_args.per_device_eval_batch_size = args.per_device_eval_batch_size
training_args.vllm_gpu_memory_utilization = 0.9
print(training_args)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    turn_reward_funcs=turn_reward_funcs,
    outcome_reward_funcs=outcome_reward_funcs,
    args=training_args,
    eval_dataset=eval_dataset,
)
trainer.control.should_evaluate = True
trainer.evaluate()
