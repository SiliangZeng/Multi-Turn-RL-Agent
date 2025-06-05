import torch

import verifiers as vf

from verifiers.tools import wiki_search
from verifiers.prompts import DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import TrivialQAToolRubric
from trl import GRPOConfig


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
trainer = "mt_grpo"  # Options: "mt_grpo", "grpo"
no_turn_reward = False
max_steps = 200
turn_advantage_coef = 1.0

# Generate run name based on trainer type
if trainer == "mt_grpo":
    run_name = f"{trainer}-coef-{turn_advantage_coef}-4-outcome-reward-2-turn-reward-max-steps-{max_steps}-{model_name.split('/')[-1].lower()}"
elif trainer == "grpo":
    if no_turn_reward:
        run_name = f"{trainer}-4-outcome-reward-no-turn-reward-max-steps-{max_steps}-{model_name.split('/')[-1].lower()}"
    else:
        run_name = f"{trainer}-4-outcome-reward-2-turn-reward-max-steps-{max_steps}-{model_name.split('/')[-1].lower()}"


training_args = GRPOConfig(
    output_dir=f"outputs/{run_name}" if run_name else None,
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=20,
    num_train_epochs=1,
    bf16=True,
    adam_beta1=0.9,
    adam_beta2=0.99,
    max_grad_norm=0.1,
    num_iterations=1,
    beta=0.04,
    max_prompt_length=2094,
    max_completion_length=4096,
    per_device_train_batch_size=1,
    num_generations=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=False,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to=None,
    reward_weights=None,
    epsilon=0.2,
)

# Disable flash attention 2 on Apple silicon
model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    use_cache=False,
    # attn_implementation="flash_attention_2",
)

# Load model and tokenizer
model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs=model_kwargs)

# Setup environment
vf_env = vf.ToolEnv(
    dataset="triviaqa",
    tools=[wiki_search],
    system_prompt=DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE,
    max_steps=2,
    use_local_model=True,
    increment_step_from_tools_only=False,
)

train_dataset = vf_env.get_dataset()

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

if trainer == "mt_grpo":
    trainer = vf.MTGRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        turn_reward_funcs=turn_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        turn_advantage_coef=turn_advantage_coef,
        args=training_args,
        train_dataset=train_dataset,
    )
elif trainer == "grpo":
    trainer = vf.GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        turn_reward_funcs=turn_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        no_turn_reward=no_turn_reward,
        args=training_args,
        train_dataset=train_dataset,
    )

trainer.train()