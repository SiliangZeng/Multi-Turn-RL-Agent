#!/bin/bash

export WANDB_API_KEY=
export WANDB_ENTITY=
export WANDB_PROJECT=

# Activate the environment
source activate verifier_env

# Default values for command-line arguments
MODEL_NAME=${1:-"Qwen/Qwen2.5-7B"}
LEARNING_RATE=${2:-"1e-6"}
NUM_GENERATIONS=${3:-"21"}
PER_DEVICE_TRAIN_BATCH_SIZE=${4:-"12"}
GRAD_ACCUM_STEPS=${5:-"4"}
NUM_ITERATIONS=${6:-"2"}
MAX_STEPS=${7:-"300"}
BETA=${8:-"0"}
TURN_ADV_COEF=${9:-"1"}

# Detect the number of GPUs on the machine
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_GPUS_MINUS_1=$((NUM_GPUS - 1))
echo "Detected ${NUM_GPUS} GPUs on the machine"

# Set hyperparameters based on GPU count
if [ ${NUM_GPUS} -eq 8 ]; then
    # Configuration for 8 GPUs
    PER_DEVICE_TRAIN_BATCH_SIZE="12"
    GRAD_ACCUM_STEPS="4"
    echo "Using 8 GPU configuration with NUM_GENERATIONS=${NUM_GENERATIONS}, PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
elif [ ${NUM_GPUS} -eq 4 ]; then
    # Configuration for 4 GPUs
    PER_DEVICE_TRAIN_BATCH_SIZE="14"
    GRAD_ACCUM_STEPS="8"
    echo "Using 4 GPU configuration with NUM_GENERATIONS=${NUM_GENERATIONS}, PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
fi

# Display configuration
echo "Using ${NUM_GPUS_MINUS_1} GPUs for training and 1 GPU for rollout generation with model ${MODEL_NAME}"
echo "Turn-level advantage coefficient: ${TURN_ADV_COEF}"

# Launch the multi-turn GRPO training
accelerate launch --config-file configs/zero3.yaml --num-processes ${NUM_GPUS_MINUS_1} \
    verifiers/examples/triviaqa_search.py \
    --model_name "${MODEL_NAME}" \
    --num_gpus ${NUM_GPUS} \
    --learning_rate ${LEARNING_RATE} \
    --num_generations ${NUM_GENERATIONS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --num_iterations ${NUM_ITERATIONS} \
    --max_steps ${MAX_STEPS} \
    --beta ${BETA} \
    --trainer "mt_grpo" \
    --turn_advantage_coef ${TURN_ADV_COEF} \