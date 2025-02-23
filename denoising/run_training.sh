#!/bin/sh
CONFIG=${1:-train_adult.yaml}
MODEL=${2:-UNet}
COUNT=${3:-1} #number of models to train (default 1)
SIM_NAME="${CONFIG%.yaml}"_"$MODEL"_$(date +'%Y-%m-%d_%H-%M')

OUTPUT_DIR=lightning_logs/"$SIM_NAME"

echo Training $COUNT Model[s]

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK train_model.sge $OUTPUT_DIR $CONFIG $MODEL

# tensorboard --logdir lightning_logs/
