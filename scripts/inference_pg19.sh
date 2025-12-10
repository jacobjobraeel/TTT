#!/bin/bash

# Configuration for "Quick Test" (Stream-to-RAM)
DATA_PATH="quick_test"
DATA_NAME="deepmind/pg19" 
DATA_CONFIG=null

# Inference Settings
SEQ_LEN=8192      # Increase to 32768 if you have enough VRAM (e.g. A100 80GB)
BS=1              # Keep batch size low for long context
TOTAL_STEPS=0     # Run through validation set once then exit (implied by eval_mode=True)

# Experiment Naming
EXP_NAME="1.3b-pg19-inference"
EXP_DIR="./experiments"

mkdir -p ${EXP_DIR}/${EXP_NAME}
chmod -R 777 ${EXP_DIR}/${EXP_NAME} 2>/dev/null || true

# Memory optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run Inference
# Note: frozen_layers=[20, 21, 22, 23] is just an example for ablation. 
# You can change this list to test different layer importance.
python3 -m ttt.train \
        --mesh_dim='!-1,1,1' \
        --dtype='fp32' \
        --eval_mode=True \
        --save_checkpoint_freq=0 \
        --save_milestone_freq=1 \
        --load_model_config="pickle::experiments/1b-TTT/metadata.pkl" \
        --load_part="trainstate_params" \
        --resume_exp_name="1b-TTT" \
        --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0, frozen_layers=[20, 21, 22, 23])" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}
