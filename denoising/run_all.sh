# runs all experiments for the paper
REPEATS=1

bash run_training.sh train_adult.yaml UNet $REPEATS

bash run_training.sh train_adult.yaml REDCNN $REPEATS

bash run_training.sh train_pediatric.yaml UNet $REPEATS

bash run_training.sh train_pediatric.yaml REDCNN $REPEATS

# bash run_training.sh train_adult_augmented.yaml UNet $REPEATS # .yaml to be defined

# bash run_training.sh train_adult_augmented.yaml REDCNN $REPEATS # .yaml to be defined