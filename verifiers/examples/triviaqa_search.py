import os
import argparse
import verifiers as vf

from verifiers.rubrics import TrivialQAToolRubric

# if os.getenv("BRAVE_API_KEY"):
#     print("Using Brave as a search engine. BRAVE_API_KEY must be set. See https://brave.com/search/api/")
#     from verifiers.tools import search_brave as search 
# else:
#     print(
#         "WARNING: Using DuckDuckGo as a search engine. \
#         This may be rate limited (which can cause training to fail). \
#         Consider setting a paid BRAVE_API_KEY (https://brave.com/search/api/) to use Brave instead."
#     )
#     from verifiers.tools import search_ddg as search 

from verifiers.tools import wiki_search as search 

from verifiers.prompts import SEARCH_FEW_SHOT

parser = argparse.ArgumentParser(description='Run TriviaQA search example')
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
args = parser.parse_args()


model_name = args.model_name
print(f"Using model: {model_name}")
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.ToolEnv(
    dataset="triviaqa",
    #few_shot=SEARCH_FEW_SHOT[0],
    tools=[search],
    max_steps=2
)

train_dataset = vf_env.get_dataset()
# train_dataset = train_dataset.select(range(200))
rubric_class = TrivialQAToolRubric()
rubric = rubric_class.get_reward_funcs()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "sz-no-tool-reward-grpo-local-wiki-search_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=args.num_gpus
)
training_args.learning_rate = args.learning_rate

# rollouts per prompt
training_args.num_generations = args.num_generations
# minibatch size per GPU
training_args.per_device_train_batch_size = args.batch_size
# batches to accumulate
training_args.gradient_accumulation_steps = args.grad_accum_steps
# steps per global batch
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

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train() 

