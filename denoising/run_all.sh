# runs all experiments for the paper
REPEATS=1
OUTPUT_DIR=lightning_logs/$(date +'%Y-%m-%d_%H-%M')_augmented_hist

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
for num in $(seq 0 0.5 1.0); do
    PROPORTION=$num
    bash run_training.sh $OUTPUT_DIR train_adult_augmented.yaml UNet $REPEATS $PROPORTION
    bash run_training.sh $OUTPUT_DIR train_adult_augmented.yaml REDCNN $REPEATS $PROPORTION
done