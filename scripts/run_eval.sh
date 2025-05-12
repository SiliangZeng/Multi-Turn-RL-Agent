#!/bin/bash

accelerate launch --config-file configs/zero3.yaml --num-processes 1 \
    verifiers/examples/triviaqa_search_eval.py \
    --model_name quanwei0/grpo-4-outcome-reward-2-turn-reward-max-steps-300-qwen2.5-7b \
    --per_device_eval_batch_size 64 \
    --subfolder "checkpoint-300" \