#! /bin/octave -qf
arg_list = argv ();
if length(arg_list) > 0
  base_directory = arg_list{1}
  if length(arg_list) > 1
    save_file = arg_list{2}
  end
else
  base_directory = '/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds';
end
if ~exist('save_file', 'var')
  save_file = 'lcd_v_diameter.csv'
end
if ~exist(base_directory)
  unzip('https://sandbox.zenodo.org/record/1213653/files/CCT189.zip')
  base_directory = 'CCT189'
end

addpath(genpath('LCD_CT'));
if is_octave
  pkg load image tablicious
end

observers = {LG_CHO_2D(),...
             NPWE_2D()};
% observers = {DOG_CHO_2D()}
% observers = {LG_CHO_2D(),...
%              DOG_CHO_2D(),...
%              GABOR_CHO_2D(),...
%              };

%% Select datasets
series_1.name = 'fbp';
series_1.dir = fullfile(base_directory);

series_2.name = 'Simple CNN MSE';
series_2.dir  = fullfile(base_directory);

series_3.name = 'Simple CNN MSE with Data Augmentation';
series_3.dir  = fullfile(base_directory);

series_list = [series_1, series_2, series_3];

ground_truth_filename = fullfile(series_1.dir, 'ground_truth.mhd')
offset = 1000;
if ~exist(ground_truth_filename, 'file')
    fname = fullfile(series_1.dir, 'diameter292mm', 'fbp');
    ground_truth = approximate_groundtruth(fname, ground_truth_filename, offset);
end
ground_truth = mhd_read_image(ground_truth_filename) - offset;

for i = 1:length(series_list)
  series = series_list(i)
  res = measure_LCD_vs_diameter(series.dir, observers, ground_truth, offset);
  if i==1
      res_table = res;
  end
  if is_octave
    res_table = vertcat(res_table, res);
  else
    res_table = cat(1, res_table, res);
  end

end
write_lcd_results(res_table, save_file)
