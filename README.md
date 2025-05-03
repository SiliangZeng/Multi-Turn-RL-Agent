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


Run MS-GRPO `bash scripts/run_ms_grpo.sh`


Run MS-PO `bash scripts/run_ms_po.sh`


RUn GRPO `bash scripts/run_grpo.sh`



## Acknowledgement

Our code implementation is built upon the open-source project [verifiers](https://github.com/willccbb/verifiers).

## Citation

If you find our work useful in your research please consider citing our paper:
```

```