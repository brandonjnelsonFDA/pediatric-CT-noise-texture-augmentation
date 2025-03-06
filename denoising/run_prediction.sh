#!/bin/bash

LOG_DIR=$1

# Use parameter expansion to extract the substring
MODEL=$(echo "$LOG_DIR" | rev | cut -d'_' -f2 | rev)
NUM_WORKERS=$(($(nproc) - 1))

# Find the best checkpoint path
BEST_CHECKPOINT=$(find "$LOG_DIR" -type f -name "*.ckpt" | sort -n | tail -n 1)

if [ -z "$BEST_CHECKPOINT" ]; then
  echo "Error: Best checkpoint not found in $LOG_DIR/checkpoints"
  exit 1
fi

echo "Best checkpoint found: $BEST_CHECKPOINT"

echo "Predicting on pediatric test data..."
python main.py predict --config predict_pediatric.yaml\
                       --model $MODEL\
                       --trainer.callbacks.output_dir $LOG_DIR/PedIQ\
                       --data.num_workers $NUM_WORKERS\
                       --ckpt_path "$BEST_CHECKPOINT"

echo "Predicting on adult test data..."
python main.py predict --config predict_adult.yaml\
                       --model $MODEL\
                       --trainer.callbacks.output_dir $LOG_DIR/MayoLDGC\
                       --data.num_workers $NUM_WORKERS\
                       --ckpt_path "$BEST_CHECKPOINT"

