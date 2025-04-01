import os
import argparse
import verifiers as vf

from verifiers.rubrics import TrivialQAToolRubric
from verifiers.trainers.ms_grpo_env_trainer import MSGRPOEnvTrainer

# 使用wiki_search作为搜索工具
from verifiers.tools import wiki_search as search

parser = argparse.ArgumentParser(description='Run TriviaQA search example with MS-GRPO')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                    help='Model name or path (default: Qwen/Qwen2.5-7B-Instruct)')
parser.add_argument('--num_gpus', type=int, default=8,
                    help='Number of GPUs to use (default: 8)')
parser.add_argument('--learning_rate', type=float, default=1e-6,
                    help='Learning rate (default: 1e-6)')
parser.add_argument('--num_generations', type=int, default=21,
                    help='Rollouts per prompt (default: 21)')
parser.add_argument('--batch_size', type=int, default=12,
                    help='Per device train batch size (default: 12)')
parser.add_argument('--grad_accum_steps', type=int, default=4,
                    help='Gradient accumulation steps (default: 4)')
parser.add_argument('--num_iterations', type=int, default=2,
                    help='Number of iterations (default: 2)')
parser.add_argument('--max_steps', type=int, default=200,
                    help='Maximum number of training steps (default: 200)')
parser.add_argument('--beta', type=float, default=0.01,
                    help='Beta parameter for KL divergence (default: 0.01)')
parser.add_argument('--step_advantage_coe', type=float, default=0.5,
                    help='Coefficient for step advantage (default: 0.5)')
args = parser.parse_args()

# 加载模型和tokenizer
model_name = args.model_name
print(f"Using model: {model_name}")
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# 创建环境
vf_env = vf.ToolEnv(
    dataset="triviaqa",
    tools=[search],
    max_steps=2
)

# 获取数据集
train_dataset = vf_env.get_dataset()

# 创建奖励函数
rubric_class = TrivialQAToolRubric()

# 将奖励函数拆分为step奖励和outcome奖励
# Step奖励函数: 仅将答案存在于搜索结果中作为step reward
step_reward_funcs = [
    rubric_class.tool_execution_reward_func,  # 答案是否存在于生成内容中
    rubric_class.exist_answer_in_search_results,  # 答案是否存在于搜索结果中
]

# Outcome奖励函数: 其余所有奖励函数作为outcome reward
outcome_reward_funcs = [
    #rubric_class.tool_execution_reward_func,  # 工具执行是否成功
    #rubric_class.exist_answer_in_search_results,  # 答案是否存在于搜索结果中
    #rubric_class.exist_answer_reward_func,  # 答案是否存在于生成内容中
    rubric_class.exact_match_reward_func,  # 答案是否精确匹配
    rubric_class.parser.get_format_reward_func(),  # XML格式是否正确
    rubric_class.parser.get_xml_reward_func(),  # XML标签是否完整
]

# 配置训练参数
#run_name = f"sz-5-outcome-reward-step-coef-{args.step_advantage_coe}-ms-grpo-updated-wiki-search_{model_name.split('/')[-1].lower()}"
run_name = f"sz-3-outcome-reward-2-step-reward-coef-{args.step_advantage_coe}-ms-grpo-updated-wiki-search_{model_name.split('/')[-1].lower()}"
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

print(f"Training configuration:")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Num GPUs: {args.num_gpus}")
print(f"  Num generations: {training_args.num_generations}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Grad accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"  Num iterations: {training_args.num_iterations}")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Beta: {training_args.beta}")
print(f"  Step advantage coefficient: {args.step_advantage_coe}")

# 使用MSGRPOEnvTrainer进行训练
trainer = MSGRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    step_reward_funcs=step_reward_funcs,
    outcome_reward_funcs=outcome_reward_funcs,
    step_advantage_coe=args.step_advantage_coe,
    args=training_args,
    train_dataset=train_dataset
)

# 开始训练
trainer.train() 