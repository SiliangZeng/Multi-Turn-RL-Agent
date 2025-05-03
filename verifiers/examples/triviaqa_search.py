import argparse
import verifiers as vf

from verifiers.tools import wiki_search
from verifiers.prompts import DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import TrivialQAToolRubric


parser = argparse.ArgumentParser(description='Run TriviaQA search example with MS-GRPO')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B", help='Model name or path (default: Qwen/Qwen2.5-7B)')
parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use (default: 8)')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate (default: 1e-6)')
parser.add_argument('--num_generations', type=int, default=21, help='Rollouts per prompt (default: 21)')
parser.add_argument('--per_device_train_batch_size', type=int, default=12, help='Per device train batch size (default: 12)')
parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps (default: 4)')
parser.add_argument('--num_iterations', type=int, default=2, help='Number of iterations (default: 2)')
parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of training steps (default: 200)')
parser.add_argument('--beta', type=float, default=0.01, help='Beta parameter for KL divergence (default: 0.01)')
parser.add_argument('--trainer', type=str, default="ms_grpo", help='Trainer type (default: ms_grpo)')
parser.add_argument('--use_step_reward', type=bool, default=True, help='Use step reward (default: True)')
parser.add_argument('--step_advantage_coef', type=float, default=0.0, help='Coefficient for step advantage (default: 0.0)')
parser.add_argument('--discount_factor', type=float, default=1.0, help='Discount factor for outcome rewards (default: 1.0)')
args = parser.parse_args()

model_name = args.model_name
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.ToolEnv(
    dataset="triviaqa",
    tools=[wiki_search],
    system_prompt=DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE,
    max_steps=2
)

train_dataset = vf_env.get_dataset()

rubric_class = TrivialQAToolRubric()

step_reward_funcs = [
    rubric_class.tool_execution_reward_func,
    rubric_class.exist_answer_in_search_results,
]

outcome_reward_funcs = [
    rubric_class.exist_answer_reward_func,
    rubric_class.exact_match_reward_func,
    rubric_class.parser.get_format_reward_func(),
    rubric_class.parser.get_xml_reward_func(),
]

if args.trainer == "ms_po":
    run_name = f"{args.trainer}-discount-factor-{args.discount_factor}-4-outcome-reward-2-step-reward-max-steps-{args.max_steps}-{model_name.split('/')[-1].lower()}"
elif args.trainer == "grpo":
    if args.use_step_reward == False:
        run_name = f"{args.trainer}-coef-{args.step_advantage_coef}-4-outcome-reward-max-steps-{args.max_steps}-{model_name.split('/')[-1].lower()}"
else:
    run_name = f"{args.trainer}-coef-{args.step_advantage_coef}-4-outcome-reward-2-step-reward-max-steps-{args.max_steps}-{model_name.split('/')[-1].lower()}"

training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=args.num_gpus
)
training_args.learning_rate = args.learning_rate
training_args.num_generations = args.num_generations
training_args.per_device_train_batch_size = args.batch_size
training_args.gradient_accumulation_steps = args.grad_accum_steps
training_args.num_iterations = args.num_iterations
training_args.max_steps = args.max_steps
training_args.beta = args.beta
# training_args.report_to = "none"

print(f"Training configuration:")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Num GPUs: {args.num_gpus}")
print(f"  Num generations: {training_args.num_generations}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Grad accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"  Num iterations: {training_args.num_iterations}")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Beta: {training_args.beta}")
print(f"  Step advantage coefficient: {args.step_advantage_coef}")

if args.trainer == "ms_grpo":
    trainer = vf.MSGRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        step_reward_funcs=step_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        use_step_reward=True,
        step_advantage_coef=args.step_advantage_coef,
        args=training_args,
        train_dataset=train_dataset
    )
elif args.trainer == "ms_po":
    trainer = vf.MSPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        step_reward_funcs=step_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        use_step_reward=True,
        discount_factor=args.discount_factor,
        args=training_args,
        train_dataset=train_dataset
    )
else: # GRPO
    trainer = vf.MSGRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        step_reward_funcs=step_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        use_step_reward=args.use_step_reward,
        step_advantage_coef=0,
        args=training_args,
        train_dataset=train_dataset
    )
trainer.train()
