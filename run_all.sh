experiment_name=adding_noise_to_labels
base_directory=/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds
notes=''

# ------------------------
results_dir=results/$(date +'%m-%d-%Y_%H-%M')_$experiment_name
mkdir -p $results_dir
results_file=$results_dir/lcd_v_diameter_results.csv

noise_patch_dir='./noise_patches'
if [ ! -d  $noise_patch_dir ]; then
echo noise patch dir not found, making one now: $noise_patch_dir
python make_noise_patches.py $base_directory/CCT189_peds_fbp
fi

python train_denoiser_with_augmentation.py

python process_CCT189.py $base_directory

octave-cli measure_LCD_diameter_dependence.m $base_directory $results_file

python task_assessments.py $results_file

python noise_assessments.py $base_directory -o $results_dir

echo Now writing summary report...
python make_summary.py $results_dir $notes
echo summary written to $results_dir/summary.pdf

# Copy original files into results dir for reproducibility
cp -v *.{sh,m,py} $resul