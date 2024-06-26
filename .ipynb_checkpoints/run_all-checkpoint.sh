experiment_name=redcnn_remove_random_noise_level
base_directory=/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation
phantom_directory=$base_directory/CCT189_peds
anthropomorphic_directory=$base_directory/anthropomorphic
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

python make_noise_patches.py --data_path $phantom_directory \
                             --save_path 'noise_patches' \
                             --patch_size $patch_size

## Model Training
# augmented
python denoising/main.py --data_path /gpfs_projects/brandon.nelson/Mayo_LDGC/images \
                         --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                         --load_mode=1 \
                         --save_path ~/Dev/PediatricCTSizeAugmentation/denoising/models/redcnn_augmented \
                         --augment=0.65
# non-augmented
python denoising/main.py --data_path /gpfs_projects/brandon.nelson/Mayo_LDGC/images \
                         --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                         --load_mode=1 \
                         --save_path ~/Dev/PediatricCTSizeAugmentation/denoising/models/redcnn
## denoising test images
python apply_denoisers.py $phantom_directory

python apply_denoisers.py $anthropomorphic_directory

python patient_images.py --output_directory $results_dir

export LD_LIBRARY_PATH=
# strange bug caused by Tensorflow need to clear this variable^, when I leave tensorflow for Pytorch and octave for python these shouldnt be issues anymore
octave-cli measure_LCD_diameter_dependence.m $phantom_directory $results_file

python task_assessments.py $results_file

python noise_assessments.py $phantom_directory \
                            --output_directory $results_dir

python methods_figures.py $phantom_directory \
                          --output_directory $results_dir \
                          --patch_size $patch_size \
                          --max_images 1000 \
                          --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                          --kernel fbp

python methods_figures.py $phantom_directory \
                          --output_directory $results_dir \
                          --patch_size $patch_size \
                          --max_images 1000 \
                          --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                          --kernel RED-CNN

python methods_figures.py $phantom_directory \
                          --output_directory $results_dir \
                          --patch_size $patch_size \
                          --max_images 1000 \
                          --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                          --kernel "RED-CNN augmented"

echo Now writing summary report...
cp -v references.bib $results_dir
python make_summary.py $results_dir "$notes" \
                        --patch_size $patch_size

echo "experiment finished at $(date -u +%T\ %D), elapsed time $(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds\n\n" >> $LOG
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' - >> $LOG
printf "*\n\n*results saved to:** $results_dir\n" >> $LOG
summary_file=$results_dir/summary.pdf
printf "**summary:** $summary_file\n\n" >> $LOG
# # Copy original files into results dir for reproducibility
# cp -v *.{sh,py,m} $results_dir
cp -v *.m $results_dir
cp -v *.sh $results_dir
cp -v *.py $results_dir

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' _ >> $LOG
printf '\n\n' >> $LOG