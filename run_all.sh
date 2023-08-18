base_directory=/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds
results_file='lcd_v_diameter_results.csv'

# ------------------------
noise_patch_dir='./noise_patches'
if [ ! -d  $noise_patch_dir ]; then
echo noise patch dir not found, making one now: $noise_patch_dir
python make_noise_patches.py $base_directory/CCT189_peds_fbp
fi

python train_denoiser_with_augmentation.py

python process_CCT189.py $base_directory

# git clone https://github.com/DIDSR/LCD_CT.git

octave-cli measure_LCD_diameter_dependence.m $base_directory $results_file

python plot_diameter_results.py $results_file