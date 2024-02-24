experiment_name=redcnn_augmented_dc_bias_removed
base_directory=/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds
# add notes written in LateX that will be added to the report and log
patch_size=64
notes='Open source REDCNN implementation (https://github.com/SSinyu/RED-CNN) with augmentation added, augmentation now uses noise patches that are histogram matched to training data so only the texture differs, not the intensity statistics. Updated subtracting the DC component from the histogram matching which added about 3 HU, this is probably minor but could influence the HU accuracy'

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

noise_patch_dir=./noise_patches/patch_size_${patch_size}x${patch_size}
if [ ! -d  $noise_patch_dir ]; then
echo noise patch dir not found, making one now: $noise_patch_dir
python make_noise_patches.py --data_path $base_directory/CCT189_peds_fbp \
                             --save_path $noise_patch_dir \
                             --patch_size $patch_size
fi

## Model Training
# augmented
python denoising/main.py --data_path /gpfs_projects/brandon.nelson/Mayo_LDGC/images \
                         --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                         --load_mode=1 \
                         --save_path ~/Dev/PediatricCTSizeAugmentation/denoising/models/redcnn_augmented \
                         --augment=1
# non-augmented
python denoising/main.py --data_path /gpfs_projects/brandon.nelson/Mayo_LDGC/images \
                         --saved_path /gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files \
                         --load_mode=1 \
                         --save_path ~/Dev/PediatricCTSizeAugmentation/denoising/models/redcnn
## denoising test images
python process_CCT189.py $base_directory

python patient_images.py --output_directory $results_dir

export LD_LIBRARY_PATH=
# strange bug caused by Tensorflow need to clear this variable^, when I leave tensorflow for Pytorch and octave for python these shouldnt be issues anymore
octave-cli measure_LCD_diameter_dependence.m $base_directory $results_file

python task_assessments.py $results_file

python noise_assessments.py $base_directory \
                            --output_directory $results_dir

python methods_figures.py $base_directory \
                          --output_directory $results_dir

echo Now writing summary report...
cp -v references.bib $results_dir
python make_summary.py $results_dir "$notes"

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