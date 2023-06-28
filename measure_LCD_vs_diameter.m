function res_table = measure_LCD_vs_diameter(base_data_folder, observers, ground_truth, offset)
  diameter_dirs = dir(fullfile(base_data_folder, 'diameter*mm'));
  n_diameters = length(diameter_dirs);

  for diam_idx=1:n_diameters
      diameter_dir=diameter_dirs(diam_idx).name;
      diameter = regexp(diameter_dir, '\d+', 'match'); diameter = str2num(diameter{:});
      recons = dir(fullfile(base_data_folder, diameter_dir));
      n_recons = length(recons);
      for recon_idx=3:n_recons
        recon = recons(recon_idx).name;
        recon_dir = recons(recon_idx).folder;
        res = measure_LCD(fullfile(recon_dir, recon), observers, ground_truth, offset);
        if is_octave
         res.recon = string(recon);
         res.diameter = diameter;
        else
          res.recon(:) = string(recon);
          res.diameter(:) = diameter;
        end
        if diam_idx == 1 && recon_idx == 3
          res_table = res;
        end
        if is_octave
          res_table = vertcat(res_table, res);
        else
          res_table = cat(1, res_table, res);
        end
      end
  end
  
end

