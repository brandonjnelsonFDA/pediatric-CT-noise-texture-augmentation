#!/bin/sh
OUTPUT_DIR=$1
CONFIG=${2:-train_adult.yaml}
MODEL=${3:-UNet}
COUNT=${4:-1} #number of models to train (default 1)
PROPORTION=${5:-0} #proportion of batch that is augmented if using AugmentedDataModule

SIM_NAME="${CONFIG%.yaml}"_"$PROPORTION"_"$MODEL"
OUTPUT_DIR="$OUTPUT_DIR"/$SIM_NAME

echo Training $COUNT Model[s]

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK train_model.sge $OUTPUT_DIR $CONFIG $MODEL $PROPORTION

# tensorboard --logdir lightning_logs/
