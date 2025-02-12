#!/bin/sh
INPUT=${1:-input.csv}
SIM_NAME=peds_noise_augmentation_$(date +'%Y-%m-%d_%H-%M')
LOG_DIR=logs/$SIM_NAME
DATA_DIR='/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/'

echo Running Model Training

qsub -N $SIM_NAME train_model.sge $LOG_DIR $DATA_DIR
# probably want to make this a task job to train 3-5 models for each case to have error bars