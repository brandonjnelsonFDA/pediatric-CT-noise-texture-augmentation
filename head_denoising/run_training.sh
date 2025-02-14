#!/bin/sh
CONFIG=${1:-'mayo_ldgc_unet.yaml'}
COUNT=${2:-1} #number of models to train (default 1)

SIM_NAME=peds_noise_augmentation_$(date +'%Y-%m-%d_%H-%M')
LOG_DIR=logs/$SIM_NAME

echo Training $COUNT Model[s]

START_TASK=1
END_TASK=$COUNT
qsub -N $SIM_NAME -t $START_TASK-$END_TASK train_model.sge $LOG_DIR $CONFIG

# tensorboard --logdir lightning_logs/
