MODEL=UNet
EXPERIMENT_NAME=train_adult_"$MODEL"_$(date +'%Y-%m-%d_%H-%M')
OUTPUT_DIR=lightning_logs/$EXPERIMENT_NAME
NUM_WORKERS=11

# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
# train on adult data
echo "Starting Training on Adult Data..."
python main.py fit     --model $MODEL\
                       --trainer.max_epochs 1\
                       --trainer.logger TensorBoardLogger\
                       --trainer.logger.save_dir $OUTPUT_DIR\
                       --data MayoLDGCDataModule\
                       --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                       --data.region chest\
                       --data.num_workers $NUM_WORKERS

# Find the best checkpoint path
BEST_CHECKPOINT=$(find "$OUTPUT_DIR" -type f -name "*.ckpt" | sort -n | tail -n 1)

if [ -z "$BEST_CHECKPOINT" ]; then
  echo "Error: Best checkpoint not found in $OUTPUT_DIR/checkpoints"
  exit 1
fi

echo "Best checkpoint found: $BEST_CHECKPOINT"

echo "Predicting on pediatric test data..."
python main.py predict --model $MODEL\
                       --trainer.callbacks DicomWriter\
                       --trainer.callbacks.write_interval epoch\
                       --trainer.callbacks.output_dir $OUTPUT_DIR/PedIQ\
                       --data PediatricIQDataModule\
                       --data.data_dir /projects01/didsr-aiml/brandon.nelson/pediatric_CT_noise_augmentation\
                       --data.num_workers $NUM_WORKERS\
                       --ckpt_path "$BEST_CHECKPOINT"

echo "Predicting on adult test data..."
python main.py predict --model $MODEL\
                       --trainer.callbacks DicomWriter\
                       --trainer.callbacks.write_interval epoch\
                       --trainer.callbacks.output_dir $OUTPUT_DIR/MayoLDGC\
                       --data MayoLDGCDataModule\
                       --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                       --data.num_workers $NUM_WORKERS\
                       --ckpt_path "$BEST_CHECKPOINT"
