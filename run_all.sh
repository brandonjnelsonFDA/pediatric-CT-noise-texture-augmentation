experiment_name=redcnn_remove_random_noise_level
base_directory=data
# add notes written in LateX that will be added to the report and log
patch_size=64
notes='Open source REDCNN implementation (https://github.com/SSinyu/RED-CNN) with augmentation added, augmentation now uses noise patches that are histogram matched to training data so only the texture differs, not the intensity statistics. Removed noise_lambda = torch.rand([1])[0].item(), now only the noise magnitude equal to training set is being added, this is a simpler training than what was'

LOG=results/results_log.md

# Do not edit below
# ------------------------------------------------------------------------------------------------------------------------------------
results_dir=results/$(date +'%m-%d-%Y_%H-%M')_$experiment_name
mkdir -p $results_dir
results_file=$results_dir/lcd_v_diameter_results.csv

printf "$(date -u +%T\ %D): $experiment_name\n" >> $LOG
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' = >> $LOG
printf '\n' >> $LOG
printf "$notes \n" >> $LOG

python notebooks/make_noise_patches.py --data_path $base_directory/pediatricIQphantoms \
                             --save_path 'noise_patches' \
                             --patch_size $patch_size

## Model Training
# augmented
python denoising/main.py --data_path data/Mayo_LDGC/images \
                         --saved_path data/Mayo_LDGC/numpy_files \
                         --load_mode=1 \
                         --save_path denoising/models/redcnn_augmented \
                         --augment=0.65
# non-augmented
python denoising/main.py --data_path data/Mayo_LDGC/images \
                         --saved_path data/Mayo_LDGC/numpy_files \
                         --load_mode=1 \
                         --save_path denoising/models/redcnn
## denoising test images
python apply_denoisers.py $base_directory/'anthropomorphic'

python apply_denoisers.py $base_directory/'anthropomorphic'