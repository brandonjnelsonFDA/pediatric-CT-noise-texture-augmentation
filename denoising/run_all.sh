# runs all experiments for the paper
REPEATS=1
OUTPUT_DIR=lightning_logs/$(date +'%Y-%m-%d_%H-%M')_augmented_hist
MODELS=("UNet" "REDCNN")

# loops through all models and configurations
for model in "${MODELS[@]}"; do
    for config in train_adult.yaml train_pediatric.yaml train_head.yaml; do
        bash run_training.sh $OUTPUT_DIR $config $model $REPEATS
    done
done

# loop through models for unavailable pediatric training data
for num in $(seq 0 0.5 1.0); do
    PROPORTION=$num
    for model in "${MODELS[@]}"; do
        bash run_training.sh $OUTPUT_DIR train_adult_augmented.yaml $model $REPEATS $PROPORTION
    done
done

# make measurements on uniform phantom
python get_experiment_metadata.py $OUTPUT_DIR | python measure.py