# runs all experiments for the paper
REPEATS=1
OUTPUT_DIR=lightning_logs/$(date +'%Y-%m-%d_%H-%M')_ldliver_norm
MODELS=("UNet" "REDCNN")

# outer loop loops through training scenarios:
# 1. train_adult.yaml (adult only training data - the typical case)
# 2. train_pediatric.yaml (pediatric only training data - the ideal case for making pediatric specific models)
# 3. train_head.yaml (adult head scans, an alternative to pediatric data is training on another anatomic region with smaller FOV, similar to peds FOV)
# 4. train_head-sim.yaml (adult head scans, an alternative to pediatric data is training on another anatomic region with smaller FOV, similar to peds FOV, but using simulated head images)

for config in train_adult.yaml train_pediatric.yaml train_head.yaml train_head-sim.yaml; do
    for model in "${MODELS[@]}"; do # inner loop loops through model architectures
        bash run_training.sh $OUTPUT_DIR $config $model $REPEATS
    done
done

# now introduce augmented training looping through different levels of augmentation controlled by PROPORTION, assumes pediatric training data unavailable
for num in $(seq 0 0.1 1.0); do
    PROPORTION=$num
    for model in "${MODELS[@]}"; do # inner loop loops through model architectures
        bash run_training.sh $OUTPUT_DIR train_adult_augmented.yaml $model $REPEATS $PROPORTION
    done
done

# make measurements on uniform phantom (when the above simulations finish, run the following:)
# python get_experiment_metadata.py $OUTPUT_DIR | python measure.py