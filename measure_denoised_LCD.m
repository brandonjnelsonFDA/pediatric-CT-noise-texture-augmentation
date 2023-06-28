% demo_05 advanced tutorial
% authors: Brandon Nelson, Rongping Zeng
%
% This demo outputs AUC curves of two recon options to show how the LCD-CT tool can be used to compare to denoising devices or recon methods
%% add relevant source files and packages
addpath(genpath('LCD_CT2/src'))
% clear all;
% close all;
clc;

if is_octave
  pkg load image tablicious
end
%% User specified parameters
% specify the `base_directory` containing images to be evaluated

observers = {LG_CHO_2D()};
% observers = {DOG_CHO_2D()}
% observers = {LG_CHO_2D(),...
%              DOG_CHO_2D(),...
%              GABOR_CHO_2D(),...
%              };

%% Select datasets
base_directory = 'data/CCT189/large_dataset';
series_1.name = 'fbp';
series_1.dir = fullfile(base_directory, series_1.name);

series_2.name = 'Simple CNN MSE'
series_2.dir  = fullfile(base_directory, 'fbp_denoised_mse');

series_3.name = 'Simple CNN VGG'
series_3.dir  = fullfile(base_directory, 'fbp_denoised_vgg');

series_4.name = 'REDCNN'
series_4.dir = fullfile(base_directory, 'DL_denoised');

series_list = [series_1, series_2, series_3, series_4];
%% Next specify a ground truth image
% This is used to determine the center of each lesion for Location Known Exactly (LKE) low contrast detection

ground_truth_fname = fullfile(base_directory,series_1.name, 'ground_truth.mhd');
offset = 1000;
ground_truth = mhd_read_image(ground_truth_fname) - offset; %need to build in offset to dataset

%% run
nreader = 10;
pct_split = 0.6
seed_split = randi(1000, nreader,1);

res_table = [];
for i = 1:length(series_list)
  series = series_list(i);
  series.res = measure_LCD(series.dir, observers, ground_truth, offset, nreader, pct_split, seed_split);
  if is_octave
    series.res.recon = series.name
  else
    series.res.recon(:) = series.name;
  end
  %% combine results
  if i == 1
    res_table = series.res
  end

  if is_octave
    res_table = vertcat(res_table, series.res);
  else
    res_table = cat(1, res_table, series.res);
  end
end
if is_octave
  res_table.recon = strvcat(res_table.recon);
end
%% save results
fname = mfilename;
output_fname = ['results_', fname(1:7), '.csv'];
write_lcd_results(res_table, output_fname)
%% plot results
set_ylim = [];

fig = figure('NumberTitle', 'off', 'Name', 'AUC vs. Dose Curves', 'Position',[10 10 1000 1400]);
plot_objs = plot_results(res_table, set_ylim)
for i = 1:length(plot_objs)
  set(plot_objs(i).legend, 'fontsize', 6)
  set(plot_objs(i).legend, 'location', 'southeast')
end
savefig('denoiser_LCD_comparison');
print(fig, 'denoiser_LCD_comparison.png', '-r300', '-dpng');
res_table

