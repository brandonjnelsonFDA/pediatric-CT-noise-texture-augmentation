base_data_folder = 'data\CCT189_peds';
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
res_table = measure_LCD_vs_diameter(base_data_folder, observers, ground_truth, offset)
write_lcd_results(res_table, 'lcd_v_diameter_results.csv')
% where's the ground truth? Need to add to the dataset
