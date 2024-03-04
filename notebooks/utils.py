import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pandas as pd

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
  elif isinstance(window, tuple):
    ww = window[0]
    wl = window[1]
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


def make_montage(meta_df:pd.DataFrame, dose:int=25, diameters:list=[35.0, 11.2], recons:list = ['fbp', 'RED-CNN', 'RED-CNN augmented'],
                 phantom:str = 'MITA-LCD', roi_diameter:float=0.3, roi_center:tuple=(256, 256),  wwwl = (80, 0)):
    """
    make image montage based on given argument parameters. Recons are plotted horizontally along the x axis while different diameters are plotted on y
    :Parameters:
        :meta_df: metadata dataframe
        :dose: dose level in percent [%]
        :diameters: phantom effective diameter in cm
    """
    
     #relative to phantom diameter (decrease below the recommended 40% diameter to fit between the inserts
    # assert(phantom in ['MITA-LCD', 'uniform'])    
    all_imgs = []
    idx = 0
    for diameter in diameters:
        recon_imgs = []
        for recon in recons:
            offset = 1000 if recon == 'fbp' else 0
            filt = (meta_df['effective diameter (cm)'] == diameter) & (meta_df['Dose [%]'] == dose) & (meta_df['phantom']==phantom)
            mhd_file = meta_df[(meta_df.recon == recon) & filt].file.item()
            recon_imgs.append(load_mhd(mhd_file).squeeze()[idx] - offset)
        all_imgs.append(recon_imgs)

   
    
    if phantom in ['MITA-LCD', 'uniform']:  
        phantom_diameter_px = get_circle_diameter(all_imgs[0][0])
    else:
        phantom_diameter_px = all_imgs[0][0].shape[0]/1.1
    
    circle_selection_diameter_px = roi_diameter*phantom_diameter_px # iec standard suggests centred circle ROI 40% of phantom diameter 
    circle_selection = circle_select(all_imgs[0][0], roi_center, r = circle_selection_diameter_px/2)

    immatrix = np.concatenate([np.concatenate(row, axis=1) for row in all_imgs], axis=0)
    ctshow(immatrix, wwwl)
    plt.colorbar(fraction=0.015, pad=0.01, label='HU')
    immatrix = np.concatenate([np.concatenate(len(all_imgs[0])*[circle_selection], axis=1) for row in all_imgs], axis=0)
    plt.imshow(immatrix, alpha=0.1, cmap='Reds')
    for didx, diam in enumerate(all_imgs):
        for ridx, recon in enumerate(diam):
            nx, ny = recon.shape
            plt.annotate(f'mean: {recon[circle_selection].mean():2.0f} HU\nstd: {recon[circle_selection].std():2.0f} HU', (nx//2 + nx*ridx, nx//2 + ny*didx), fontsize=6, bbox=dict(boxstyle='square,pad=0.3', fc="lightblue", ec="steelblue"))
    plt.title(' | '.join(recons))
    plt.ylabel(' mm |'.join(map(lambda o: str(o), diameters)) + ' mm')