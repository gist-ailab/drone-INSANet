#!/bin/bash
cd "$(dirname "$0")/.."

# e.g. If you want to use a single GPU:
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 python train_eval_ours.py \
    --exp 'AttNet'
    
# # e.g. If you want to use multiple GPUs:
# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python train_eval.py \
#     --exp 'INSANet'
