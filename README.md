# Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment

This repository contains the code of RoSTE introduced in our work: ["Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"]()

## Installation

Install Python environment
```bash
conda create -n verifier_env python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verifier_env

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Install Java environment (if needed)
```bash
apt-get remove --purge openjdk* -y

apt-get update
apt-get install -y openjdk-21-jdk

java -version
```

You can run `bash quick_start_install.sh` for a quick start installation.

Download Wiki search index 
```bash
python verifiers/tools/local_wiki_search.py
```

## Search Agent Overview

Tool Env: `verifiers/envs/tool_env.py`

Search Tool: `verifiers/tools/local_wiki_search.py`

System Prompt: `verifiers/prompts/system_prompts.py`

Trainers:
- MS-GRPO: `verifiers/trainer/ms_grpo_env_trainer.py`
- MS-PO: `verifiers/trainer/ms_po_env_trainer.py`

Reward Functions `verifiers/rubric/triviaqa_rubric.py`
- Step-Level Rewards:
    - `tool_execution_reward_func`
    - `exist_answer_in_search_results`
- Outcome-Level Rewards:
    - `exist_answer_reward_func`
    - `exact_match_reward_func`
    - `parser.get_format_reward_func`
    - `parser.get_xml_reward_func`



## Usage


Run MS-GRPO
```bash
# `bash scripts/run_ms_grpo.sh`
accelerate launch --config-file configs/zero3.yaml --num-processes 7 \
    verifiers/examples/triviaqa_search.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --num_gpus 8 \
    --learning_rate 1e-6 \
    --num_generations 21 \
    --per_device_train_batch_size 12 \
    --grad_accum_steps 4 \
    --num_iterations 2 \
    --max_steps 300 \
    --beta 0 \
    --trainer "ms_grpo" \
    --step_advantage_coef 1 \
```

Run MS-PO `bash scripts/run_ms_po.sh`
```bash
# `bash scripts/run_ms_po.sh`
accelerate launch --config-file configs/zero3.yaml --num-processes 7 \
    verifiers/examples/triviaqa_search.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --num_gpus 8 \
    --learning_rate 1e-6 \
    --num_generations 21 \
    --per_device_train_batch_size 12 \
    --grad_accum_steps 4 \
    --num_iterations 2 \
    --max_steps 300 \
    --beta 0 \
    --trainer "ms_po" \
    --discount_factor 1 \
```

Run GRPO with Step-Level Rewards
```bash
# bash scripts/run_grpo.sh
accelerate launch --config-file configs/zero3.yaml --num-processes 7 \
    verifiers/examples/triviaqa_search.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --num_gpus 8 \
    --learning_rate 1e-6 \
    --num_generations 21 \
    --per_device_train_batch_size 12 \
    --grad_accum_steps 4 \
    --num_iterations 2 \
    --max_steps 300 \
    --beta 0 \
    --trainer "grpo" \
    --use_step_reward True \
```

RUn GRPO without Step-Level Rewards
```bash
# bash scripts/run_grpo.sh
accelerate launch --config-file configs/zero3.yaml --num-processes 7 \
    verifiers/examples/triviaqa_search.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --num_gpus 8 \
    --learning_rate 1e-6 \
    --num_generations 21 \
    --per_device_train_batch_size 12 \
    --grad_accum_steps 4 \
    --num_iterations 2 \
    --max_steps 300 \
    --beta 0 \
    --trainer "grpo" \
    --use_step_reward False \
```

## Acknowledgement

Our code implementation is built upon the open-source project [verifiers](https://github.com/willccbb/verifiers).

## Citation

If you find our work useful in your research please consider citing our paper:
```

```