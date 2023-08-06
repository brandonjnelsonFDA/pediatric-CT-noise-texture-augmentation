%base_data_folder = 'data\CCT189_peds';
base_data_folder = 'D:\Dev\Datasets\CCT189_CT_sims\CCT189_peds';
if ~exist(base_data_folder)
  unzip('https://sandbox.zenodo.org/record/1213653/files/CCT189.zip')
end

addpath(genpath('LCD_CT'));
observers = {LG_CHO_2D()};
% observers = {DOG_CHO_2D()}
% observers = {LG_CHO_2D(),...
%              DOG_CHO_2D(),...
%              GABOR_CHO_2D(),...
%              };

%% Select datasets
base_directory = 'data';
base_directory = 'D:\Dev\Datasets\CCT189_CT_sims'
series_1.name = 'fbp';
series_1.dir = fullfile(base_directory, 'CCT189_peds');

series_2.name = 'Simple CNN MSE'
series_2.dir  = fullfile(base_directory, 'CCT189_peds_denoised_mse');

series_3.name = 'Simple CNN VGG'
series_3.dir  = fullfile(base_directory, 'CCT189_peds_denoised_vgg');

series_list = [series_1, series_2, series_3];

ground_truth_filename = fullfile(base_data_folder, 'ground_truth.mhd');
offset = 1000;
if ~exist(ground_truth_filename, 'file')
    fname = fullfile(base_data_folder, 'diameter292mm', 'fbp');
    ground_truth = approximate_groundtruth(fname, ground_truth_filename, offset);
end
ground_truth = mhd_read_image(ground_truth_filename) - offset;

for i = 1:length(series_list)
  series = series_list(i);
  res = measure_LCD_vs_diameter(series.dir, observers, ground_truth, offset);
  if i==1
      res_table = res;
  end
  res_table = cat(1, res_table, res);

end
write_lcd_results(res_table, 'lcd_v_diameter_results.csv')
% where's the ground truth? Need to add to the dataset
