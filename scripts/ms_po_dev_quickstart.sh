#!/bin/bash

MODEL_NAME=${1:-"Qwen/Qwen2.5-7B"}
LEARNING_RATE=${2:-"1e-6"}
GRAD_ACCUM_STEPS=${5:-"4"}
NUM_ITERATIONS=${6:-"2"}
MAX_STEPS=${7:-"200"}
BETA=${8:-"0"}

# Get the number of GPUs on the machine
source activate verifier_env

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_GPUS_MINUS_1=$((NUM_GPUS - 1))
echo "Detected ${NUM_GPUS} GPUs on the machine"

# Set hyperparameters based on GPU count
if [ ${NUM_GPUS} -eq 8 ]; then
    # Hyperparameters for 8 GPUs
    NUM_GENERATIONS=${3:-"21"}
    BATCH_SIZE=${4:-"12"}
    echo "Using 8 GPU configuration with NUM_GENERATIONS=${NUM_GENERATIONS}, BATCH_SIZE=${BATCH_SIZE}"
elif [ ${NUM_GPUS} -eq 4 ]; then
    # Hyperparameters for 4 GPUs
    NUM_GENERATIONS=${3:-"21"}
    BATCH_SIZE=${4:-"14"}
    echo "Using 4 GPU configuration with NUM_GENERATIONS=${NUM_GENERATIONS}, BATCH_SIZE=${BATCH_SIZE}"
fi

echo "Using ${NUM_GPUS_MINUS_1} GPUs for MS-GRPO training with model ${MODEL_NAME}"

# Launch the multi-step GRPO training
accelerate launch --config-file configs/zero3.yaml --num-processes ${NUM_GPUS_MINUS_1} \
  verifiers/examples/triviaqa_search_ms_po_dev.py \
  --model_name "${MODEL_NAME}" \
  --num_gpus ${NUM_GPUS} \
  --learning_rate ${LEARNING_RATE} \
  --num_generations ${NUM_GENERATIONS} \
  --batch_size ${BATCH_SIZE} \
  --grad_accum_steps ${GRAD_ACCUM_STEPS} \
  --num_iterations ${NUM_ITERATIONS} \
  --max_steps ${MAX_STEPS} \
  --beta ${BETA} \