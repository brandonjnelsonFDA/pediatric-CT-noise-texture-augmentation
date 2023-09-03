experiment_name=112-292mm_adding_to_label_0.3thresh
base_directory=/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds
# add notes written in LateX that will be added to the report and log
notes='From experimenting I found a threshold of 0.3 matched the noise level in the low noise target image.
This seems like good justifaction for the choise in value and the noise texture looks good in the notebook.

There is some blurring compared to the target but its qualitatively better than base MSE, lets see how the 
full evaluation turns out.

This experiment is also using the built in keras model.fit rather than the manually written loop from before'
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

export LD_LIBRARY_PATH=
# strange bug caused by Tensorflow need to clear this variable^, when I leave tensorflow for Pytorch and octave for python these shouldnt be issues anymore
octave-cli measure_LCD_diameter_dependence.m $base_directory $results_file

python task_assessments.py $results_file

python noise_assessments.py $base_directory -o $results_dir

python methods_figures.py $base_directory -o $results_dir

echo Now writing summary report...
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