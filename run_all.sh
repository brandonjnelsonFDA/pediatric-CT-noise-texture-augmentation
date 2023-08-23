experiment_name=adding_noise_to_labels_151mm_only
base_directory=/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds
# add notes written in LateX that will be added to the report and log
notes='In the previous experiment I was adding 151mm noise to the *image*, rather than the *label* (what I said I was going to do). 
So I corrected the file names to reflect this and am rerunning this experiment by adding noise to the label `image = label + noise_lambda[0]*noise_patch` [line 91] of `train_denoiser_with_augmentation.py`

If this continues to have minimal effect I will explore increasing the magnitude of added noise, by increasing the likihood of adding noise (raising the threshold of add_noise line 89)'
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

noise_patch_dir='./noise_patches'
if [ ! -d  $noise_patch_dir ]; then
echo noise patch dir not found, making one now: $noise_patch_dir
python make_noise_patches.py $base_directory/CCT189_peds_fbp
fi
export TF_CPP_MIN_LOG_LEVEL=2 #<https://github.com/tensorflow/tensorflow/issues/59779>
python train_denoiser_with_augmentation.py

python process_CCT189.py $base_directory

octave-cli measure_LCD_diameter_dependence.m $base_directory $results_file

python task_assessments.py $results_file

python noise_assessments.py $base_directory -o $results_dir

echo Now writing summary report...
python make_summary.py $results_dir "$notes"
duration=$SECONDS
echo "experiment finished at $(date -u +%T\ %D), elapsed time $(($duration / 60)) minutes and $(($duration % 60)) seconds\n\n" >> $LOG
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