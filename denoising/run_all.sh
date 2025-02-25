# runs all experiments for the paper
REPEATS=1
OUTPUT_DIR=lightning_logs/augmented_$(date +'%Y-%m-%d_%H-%M')

# the typical case where you have access to an adult dataset
bash run_training.sh $OUTPUT_DIR train_adult.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_adult.yaml REDCNN $REPEATS

#ideal case when there's access to pediatric data
bash run_training.sh $OUTPUT_DIR train_pediatric.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_pediatric.yaml REDCNN $REPEATS

# does training on a different small anatomy have similar results to dedicated pediatric model?
bash run_training.sh $OUTPUT_DIR train_head.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_head.yaml REDCNN $REPEATS

# does training on pediatric sized phantoms have similar results to dedicated pediatric model? (both alternatives when pediatric training data not available)
bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.1.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.1.yaml REDCNN $REPEATS

bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.3.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.3.yaml REDCNN $REPEATS

bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.5.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.5.yaml REDCNN $REPEATS

bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.7.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.7.yaml REDCNN $REPEATS

bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.9.yaml UNet $REPEATS
bash run_training.sh $OUTPUT_DIR train_adult_augmented_0.9.yaml REDCNN $REPEATS