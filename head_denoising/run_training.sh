#!/bin/sh
INPUT=${1:-input.csv}
SIM_NAME=peds_noise_augmentation_$(date +'%Y-%m-%d_%H-%M')
LOG_DIR=logs/$SIM_NAME
DATA_DIR='/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/'
COUNT=3

echo Training $COUNT Model[s]

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK train_model.sge $LOG_DIR $DATA_DIR $COUNT

tensorboard --logdir lightning_logs/
