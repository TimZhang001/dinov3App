#!/bin/bash
# DINOv3 Classification Training Script

# Configuration
DATA_DIR="dataset/classification/abnormal26/abnormal_dataset"
OUTPUT_DIR="outputs/classifier_vits16"
MODEL_TYPE="vits16"

# Training parameters
BATCH_SIZE=64
LR=0.001
EPOCHS=100
RESIZE=256

# Run training
python train.py \
    --data-dir "$DATA_DIR" \
    --model-type "$MODEL_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --resize $RESIZE \
    --num-workers 8 \
    --device cuda
