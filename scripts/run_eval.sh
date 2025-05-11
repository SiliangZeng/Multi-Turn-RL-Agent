#!/bin/bash


accelerate launch --config-file configs/zero3.yaml --num-processes 1 \
    verifiers/examples/triviaqa_search_eval.py \
    --model_name Qwen/Qwen2.5-7B \
