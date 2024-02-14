# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
test_input = np.load('data/Denoising_Data/test_input.npy')
test_target = np.load('data/Denoising_Data/test_target.npy')

# load and denoising pediatric XCAT patients and examples from LDGC dataset

base_denoising_model = tf.keras.models.load_model('models/simple_cnn_denoiser')
aug_denoising_model = tf.keras.models.load_model('models/simple_cnn_denoiser_augmented')
# %%
def ctshow(img, window='soft_tissue'):
  # Define some specific window settings here
  if window == 'soft_tissue':
    ww = 400
    wl = 40
  elif window == 'bone':
    ww = 2500
    wl = 480
  elif window == 'lung':
    ww = 1500
    wl = -600
  else:
    ww = 6.0 * img.std()
    wl = img.mean()

  # Plot image on clean axes with specified window level
  vmin = wl - ww // 2
  vmax = wl + ww // 2
  plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
  plt.xticks([])
  plt.yticks([])

  return

# %%
nexample = 2
base_CNNout = base_denoising_model.predict(test_input, batch_size=1)
aug_CNNout = aug_denoising_model.predict(test_input, batch_size=1)
# %%

plt.figure(figsize=(16, 8), tight_layout=True)

plt.subplot(2, 2, 1)
plt.title('Low Dose Input', fontsize=16)
ctshow(test_input[3, 110:-110, 50:-50, 0])

plt.subplot(2, 2, 2)
plt.title('Hi Dose Input', fontsize=16)
ctshow(test_target[3, 110:-110, 50:-50, 0])

plt.subplot(2, 2, 3)
plt.title('Base Denoiser', fontsize=16)
ctshow(base_CNNout[3, 110:-110, 50:-50, 0])

plt.subplot(2, 2, 4)
plt.title('Denoiser with Augmentation', fontsize=16)
ctshow(aug_CNNout[3, 110:-110, 50:-50, 0])
plt.savefig(f'test_patient.png', dpi=600, bbox_inches='tight')
# %% Pediatric XCAT Example
from pathlib import Path
import numpy as np

infant_dir = Path('/gpfs_projects/brandon.nelson/DLIR_Ped_Generalizability/geometric_phantom_studies/main/anthropomorphic/simulations/male_infant_ref_atn_1/monochromatic/diameter111mm/I0_0030000/fbp_sharp')
adult_dir = Path('/gpfs_projects/brandon.nelson/DLIR_Ped_Generalizability/geometric_phantom_studies/main/anthropomorphic/simulations/male_pt148_atn_1/monochromatic/diameter342mm/I0_0030000/fbp_sharp')
# %%
infant_imfile = list(infant_dir.rglob('*.raw'))[0]
adult_imfile = list(adult_dir.rglob('*.raw'))[0]
# %%
infant_img = np.fromfile(infant_imfile, dtype='int16').reshape(1, 512,512) - 1024
adult_img = np.fromfile(adult_imfile, dtype='int16').reshape(1, 512,512) - 1024
# %%
ctshow(infant_img[0])
# %%
ctshow(adult_img[0])
# %%
infant_base_denoised = base_denoising_model.predict(infant_img, batch_size=1)
infant_aug_denoised = aug_denoising_model.predict(infant_img, batch_size=1)

adult_base_denoised = base_denoising_model.predict(adult_img, batch_size=1)
adult_aug_denoised = aug_denoising_model.predict(adult_img, batch_size=1)
# %%
ctshow(infant_base_denoised[0])
# %%
ctshow(infant_aug_denoised[0])
# %%
