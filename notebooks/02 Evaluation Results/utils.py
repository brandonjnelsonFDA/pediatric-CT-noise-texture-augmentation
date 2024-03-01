import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

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

def circle_select(img, xy, r):
    assert(img.ndim == 2)
    circle_mask = np.zeros_like(img)
    for i in range(circle_mask.shape[0]):
        for j in range(circle_mask.shape[1]):
            if (i-xy[0])**2 + (j-xy[1])**2 < r**2:
                 circle_mask[i,j] = True
    return circle_mask.astype(bool)

def get_circle_diameter(img):
    """Assumes an image of a uniform water phantom that can be easily segmented using a mean intensity threshold"""
    return 2*np.sqrt((img > img.mean()).sum()/np.pi)  #A = pi r^2 --> r = sqrt(A/pi) --> d = 2*r = 2*sqrt(A/pi)

def load_mhd(mhd_file):
    """meta header file, see examples here: <https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html>"""
    return sitk.GetArrayFromImage(sitk.ReadImage(mhd_file))