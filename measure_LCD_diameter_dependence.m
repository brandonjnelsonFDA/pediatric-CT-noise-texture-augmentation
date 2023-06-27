base_data_folder = 'data\CCT189_peds';
if ~exist(base_data_folder)
  unzip('https://sandbox.zenodo.org/record/1213653/files/CCT189.zip?download=1', fullfile('data', 'CCT189_peds'))
end

addpath(genpath('LCD_CT2'));
observers = {LG_CHO_2D()};
% observers = {DOG_CHO_2D()}
% observers = {LG_CHO_2D(),...
%              DOG_CHO_2D(),...
%              GABOR_CHO_2D(),...
%              };

diameter_dirs = dir(fullfile(base_data_folder, 'diameter*mm'));
n_diameters = length(diameter_dirs);

ground_truth_filename = fullfile(base_data_folder, 'diameter112mm', 'fbp');
offset = 1000;
image_dir = fullfile(base_data_folder, 'diameter112mm', 'fbp');
ground_truth = approximate_groundtruth(image_dir, ground_truth_filename, offset)
ground_truth = mhd_read_image(ground_truth_filename) - offset;

for diam_idx=1:n_diameters
    diameter_dir=diameter_dirs(diam_idx).name;
    diameter = regexp(diameter_dir, '\d+', 'match'); diameter = str2num(diameter{:});
    recons = dir(fullfile(base_data_folder, diameter_dir));
    n_recons = length(recons);
    for recon_idx=3:n_recons
      recon = recons(recon_idx).name;
      res = measure_LCD(fullfile(diameter_dir, recon), observers, ground_truth, offset);
      if is_octave
       res.recon = recon;
       res.diameter = diameter;
      else
        res.recon(:) = recon;
        res.diameter(:) = diameter;
      end
      if i == 3
        res_table = res
      end
      if is_octave
        res_table = vertcat(res_table, res);
      else
        res_table = cat(1, res_table, res);
      end
    end
end

% where's the ground truth? Need to add to the dataset
