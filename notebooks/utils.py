import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
from skimage.transform import resize

from PIL import Image

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

def get_ground_truth(fname):
    fname = Path(fname)
    if fname.stem.startswith('signal'):
        gt_file = 'noise_free.mhd'
        return Path(fname).parents[2] / gt_file
    if fname.stem.startswith('ACR464'):
        gt_file = 'true.mhd'
        return Path(fname).parents[3] / gt_file
    else:
        gt_file = 'true.mhd'
        return Path(fname).parents[2] / gt_file

def center_crop(img, thresh=-950):
    """square cropping where side length is based upon thresholded avergae *width*,
    assuming the imaged object is wider (covers more columns) than tall (covers fewer rows)""" 
    img_crop = img[:, img.mean(axis=0) > thresh]
    img_crop = img_crop[img.mean(axis=0) > thresh, :]
    return img_crop

def center_crop_like(other, ref, thresh=-950):
    """square cropping where side length is based upon thresholded avergae *width*,
    assuming the imaged object is wider (covers more columns) than tall (covers fewer rows)""" 
    img_crop = other[:, ref.mean(axis=0) > thresh]
    img_crop = img_crop[ref.mean(axis=0) > thresh, :]
    return img_crop

def wwwl_to_minmax(wwwl:tuple): return wwwl[1] - wwwl[0]/2, wwwl[1] + wwwl[0]/2

def make_montage(meta_df:pd.DataFrame, dose:int=25, fovs:list=[25.0, 15.0], recons:list = ['fbp', 'RED-CNN', 'RED-CNN augmented'],
                 phantom:str='ACR464', roi_diameter:float|int=0.4, roi_center:tuple|str=(256, 256), wwwl=(80, 0), crop_to_fit=True, figure=None, fontsize=6, axis=None):
    """
    make image montage based on given argument parameters. Recons are plotted horizontally along the x axis while different diameters are plotted on y
    :Parameters:
        :meta_df: metadata dataframe
        :dose: dose level in percent [%]
        :fovss: FOV in cm
        :recons: list of recon kernels to display on the horizontal x-axis
        :phantom: which image phantom is expected ['MITA-LCD', 'uniform', 'anthropomorphic', 'ACR464'], it should be in meta_df
        :roi_diameter: diameter of circlular ROI. If `roi_diameter` is an **integer** then roi_diameter is in pixels e.g. 100, else if `roi_diameter` is **float** e.g. 0.3 then it is assumed a fraction of the phantom's effective diameter. Note for noise measurements IEC standard suggests centred circle ROI 40% of phantom diameter
        :roi_center: xy coordinates for roi center, or organ e.g. liver, if phantom = 'anthropomorphic'. If `str` is provided an is in the organ list `['liver']` a **random** centered roi that fits in the organ will be provided. If `roi_center` is a tuple, e.g. (256, 256) then an roi at those exact coordinates will be given
        :wwwl: window width and window level display settings, examples: soft tissues (400, 50), liver (150, 30) for recommend display settings see <https://radiopaedia.org/articles/windowing-ct?lang=us> for more information
    """
    if (axis is None) or (figure is None):
        figure, axis = plt.subplots()
    all_imgs = []
    all_gts = []
    circle_selections = []
    idx = 0
    for fov in fovs:
        recon_imgs = []
        recon_gts = []
        recon_selections = []
        phantom_df =  meta_df[(meta_df.phantom==phantom)]
        available_fovs = sorted(phantom_df['FOV [cm]'].unique())
        if fov not in available_fovs:
            print(f'FOV {fov} not in {available_fovs}')
            return
        phantom_df = phantom_df[phantom_df['FOV [cm]']==fov]
        available_doses = sorted(phantom_df['Dose [%]'].unique())
        if dose not in available_doses:
            print(f'dose {dose}% not in {available_doses}')
            return
        phantom_df = phantom_df[phantom_df['Dose [%]']==dose]

        selection = None
        for recon in recons:
            nfiles = len(phantom_df[phantom_df.recon == recon])
            if nfiles > 1:
                print(f'{nfiles} files found, taking first')
                mhd_file = phantom_df[phantom_df.recon == recon].iloc[0].file
            elif nfiles == 1:
                mhd_file = phantom_df[phantom_df.recon == recon].file.item()
            else:
                raise RuntimeError('No files found')
            img = load_mhd(mhd_file).squeeze()[idx]
            gt = load_mhd(get_ground_truth(mhd_file))
            
            if crop_to_fit:
                original_shape = gt.shape
                img = center_crop_like(img, gt)
                gt = center_crop(gt)
                img = resize(img, original_shape, anti_aliasing=True)
                gt = resize(gt, original_shape, anti_aliasing=True)
                
            if phantom in ['MITA-LCD', 'uniform', 'ACR464']:  
                phantom_diameter_px = get_circle_diameter(gt)
            else:
                phantom_diameter_px = gt.shape[1]/1.1
            if (phantom == 'ACR464') & (fov < 20):
                phantom_diameter_px = gt.shape[1] * 20/fov
            
            circle_selection_diameter_px = roi_diameter if isinstance(roi_diameter, int) else roi_diameter*phantom_diameter_px
            if isinstance(roi_center, tuple | list):
                selection = circle_select(gt, roi_center, r = circle_selection_diameter_px/2)
            elif isinstance(roi_center, str):
                available_organs = display_settings
                if roi_center not in available_organs: raise ValueError(f'roi center {roi_center} not in {available_organs}')
                hu_min, hu_max = wwwl_to_minmax(available_organs[roi_center])
                if selection is None:
                    selection = add_random_circle_lesion(gt, mask=((gt >= hu_min) & (gt < hu_max)), radius=circle_selection_diameter_px/2)[1].astype(bool)
            
            recon_imgs.append(img)
            recon_gts.append(gt)
            recon_selections.append(selection)

        all_imgs.append(recon_imgs)
        all_gts.append(recon_gts)
        circle_selections.append(recon_selections)
    
    immatrix = np.concatenate([np.concatenate(row, axis=1) for row in all_imgs], axis=0)
    if isinstance(wwwl, str):
        if wwwl not in display_settings:
            raise ValueError(f'{wwwl} not in {display_settings}')
        wwwl = display_settings[wwwl]
    ctshow(immatrix, wwwl)
    immatrix = np.concatenate([np.concatenate(row, axis=1) for row in circle_selections], axis=0)
    axis.imshow(immatrix, alpha=0.1, cmap='Blues')
    plt.colorbar(ax=axis, fraction=0.015, pad=0.01, label='HU')
    ylvl = 60
    for didx, diam in enumerate(all_imgs):
        for ridx, recon in enumerate(diam):
            nx, ny = recon.shape
            axis.annotate(f'mean: {recon[circle_selections[didx][ridx]].mean():2.0f} HU\nstd: {recon[circle_selections[didx][ridx]].std():2.0f} HU',
                         (ny//2 + ny*ridx, nx//2 + nx*didx), fontsize=fontsize, bbox=dict(boxstyle='square,pad=0.3', fc="lightblue", ec="steelblue"))

        eff_diam_cm = meta_df[meta_df['FOV [cm]'] == fovs[didx]]['effective diameter [cm]'].iloc[0]
        pix_size = fov/nx
        eff_diam_px = np.ceil(eff_diam_cm/pix_size)
        subgroup = meta_df[meta_df['effective diameter [cm]'] == eff_diam_cm]['pediatric subgroup'].iloc[0]
        axis.annotate(f'{subgroup}\n{eff_diam_cm} cm', xy=(10, ylvl), xytext=(256, ylvl), bbox=dict(boxstyle='square,pad=0.3', fc="white", ec="black"), horizontalalignment='center', fontsize=fontsize)
        profile = np.where(recon.mean(axis=0)>-950)[0]
        x0, x1 = profile[0], profile[-1]
        axis.annotate('', xy=(x0, ylvl+18), xytext=(x1, ylvl+18), arrowprops=dict(arrowstyle='<->'))
        ylvl+=ny
    str_lens = [len(o) for o in recons]
    max_len = max(str_lens)
    recon_str = ' | '.join([(max_len-len(o))//2*' '+o+(max_len-len(o))//2*' ' for o in recons])
    axis.set_title('Recon\n'+ recon_str)
    axis.set_ylabel('FOV\n'+' cm | '.join(map(lambda o: str(round(o)), fovs[::-1])) + ' cm')
    # add scalebar
    return figure, axis
    
# from https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/draw/draw.py#L11 

def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """Generate coordinates of points within ellipse bounded by shape.

    Parameters
    ----------
    shape :  iterable of ints
        Shape of the input image.  Must be at least length 2. Only the first
        two values are used to determine the extent of the input image.
    center : iterable of floats
        (row, column) position of center inside the given shape.
    radii : iterable of floats
        Size of two half axes (for row and column)
    rotation : float, optional
        Rotation of the ellipse defined by the above, in radians
        in range (-PI, PI), in contra clockwise direction,
        with respect to the column-axis.

    Returns
    -------
    rows : iterable of ints
        Row coordinates representing values within the ellipse.
    cols : iterable of ints
        Corresponding column coordinates representing values within the ellipse.
    """
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)


def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.

    Parameters
    ----------
    r, c : double
        Centre coordinate of ellipse.
    r_radius, c_radius : double
        Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output pixel
        coordinates. This is useful for ellipses which exceed the image size.
        By default the full extent of the ellipse are used. Must be at least
        length 2. Only the first two values are used to determine the extent.
    rotation : float, optional (default 0.)
        Set the ellipse rotation (rotation) in range (-PI, PI)
        in contra clock wise direction, so PI/2 degree means swap ellipse axis

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of ellipse.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Examples
    --------
    >>> from skimage.draw import ellipse
    >>> img = np.zeros((10, 12), dtype=np.uint8)
    >>> rr, cc = ellipse(5, 6, 3, 5, rotation=np.deg2rad(30))
    >>> img[rr, cc] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    Notes
    -----
    The ellipse equation::

        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1


    Note that the positions of `ellipse` without specified `shape` can have
    also, negative values, as this is correct on the plane. On the other hand
    using these ellipse positions for an image afterwards may lead to appearing
    on the other side of image, because ``image[-1, -1] = image[end-1, end-1]``

    >>> rr, cc = ellipse(1, 2, 3, 6)
    >>> img = np.zeros((6, 12), dtype=np.uint8)
    >>> img[rr, cc] = 1
    >>> img
    array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]], dtype=uint8)
    """

    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
                   + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
                   + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


def add_random_circle_lesion(image, mask, radius=20, contrast=-100):
    """
    Returns tuple of (image with lesion, lesion only image, lesion coordinate (x,y))
    :tol: tolerance of fitting lesion into mask, smaller values increases the requirement ensuring more of the circle overlaps with the mask
    """
    r = radius
    lesion_image = np.zeros_like(image)
    x, y = np.argwhere(mask)[np.random.randint(0, mask.sum())]
    rr, cc = ellipse(x, y, r, r, shape=lesion_image.shape)
    lesion_image[rr, cc] = contrast #in HU

    counts = 0
    while np.sum(mask & (lesion_image==contrast)) < np.sum(lesion_image==contrast): #can increase threshold to size of lesion
        counts += 1
        lesion_image = np.zeros_like(image)

        x, y = np.argwhere(mask)[np.random.randint(0, mask.sum())]
        rr, cc = ellipse(x, y, r, r, shape=lesion_image.shape)
        lesion_image[rr, cc] = contrast #in HU
        max_attempts = 1000
        if counts > max_attempts:
            raise ValueError(f"Failed to insert lesion into mask after {max_attempts} attempts exceeded")

    img_w_lesion = image + lesion_image
    return img_w_lesion, lesion_image, (x, y)

def noise_reduction(fbp_std, denoised_std): return 100*(fbp_std - denoised_std)/fbp_std

def measure_roi_std_results(meta_df, roi_diameter=None):
    """
    :Parameters:
        :meta_df: metadata dataframe
        :roi_diameter: diameter of circlular ROI. If `roi_diameter` is an **integer** then roi_diameter is in pixels e.g. 100, else if `roi_diameter` is **float** e.g. 0.3 then it is assumed a fraction of the phantom's effective diameter. Note for noise measurements IEC standard suggests centred circle ROI 40% of phantom diameter
    """
    meta_df = meta_df.copy()
    meta_df['sim number'] = np.nan
    meta_df['noise std'] = np.nan
    default_diameters = {'uniform': 0.4, 'MITA-LCD':0.3, 'anthropomorphic': 0.2, 'ACR464': 0.4}
    if roi_diameter:
        if isinstance(roi_diameter, dict):
            default_diameters.update(roi_diameter)
        else:
            default_diameters = {k: roi_diameter for k in default_diameters}
    roi_diameter = default_diameters
    rows_list = []
    for idx, patient in meta_df.iterrows():
        if idx % (len(meta_df) // 10) == 0:
            print(idx,'/',meta_df.shape[0])
        img = load_mhd(patient.file)
        if img.ndim == 2: img = img[None, ...]
        gt = load_mhd(get_ground_truth(patient.file))
        
        if patient.phantom in ['uniform', 'MITA-LCD', 'ACR464']:
            if (patient.phantom == 'ACR464') & (patient['FOV [cm]'] < 20):
                 phantom_diameter_px = gt.shape[1] * 20/patient['FOV [cm]']
            phantom_diameter_px = get_circle_diameter(gt)
            circle_selection_diameter_px = roi_diameter[patient.phantom]*phantom_diameter_px # iec standard suggests centred circle ROI 40% of phantom diameter 
            circle_selection = circle_select(gt, xy=(gt.shape[0]//2, gt.shape[1]//2), r = circle_selection_diameter_px/2)
        else:
            phantom_diameter_px = gt.shape[-1]/1.1 #assumes fov size of 110% effective diameter
            circle_selection_diameter_px = roi_diameter[patient.phantom]*phantom_diameter_px
            try:
                circle_selection = add_random_circle_lesion(img[0], mask=((gt >= 50) & (gt < 100)), radius=circle_selection_diameter_px/2)[1].astype(bool)
            except:
                circle_selection = None
                print(f"Warning: failed to insert lesion of diameter {circle_selection_diameter_px} into ground truth, considering using smaller `roi_diameter` current = {roi_diameter}")
            
        for n, o in enumerate(img):
            patient['sim number'] = n
            patient['noise std'] = o[circle_selection].std() if circle_selection is not None else np.nan
            rows_list.append(pd.DataFrame(patient).T)
    return pd.concat(rows_list, ignore_index=True)

def rmse(x ,y): return np.sqrt(np.mean((x.ravel()-y.ravel())**2))


def measure_rmse_results(meta_df):
    meta_df = meta_df.copy()
    meta_df['sim number'] = np.nan
    meta_df['rmse'] = np.nan
    rows_list = []
    for idx, patient in meta_df.iterrows():
        if idx % (len(meta_df) // 10) == 0:
            print(idx,'/',meta_df.shape[0])
        img = load_mhd(patient.file)
        if img.ndim == 2: img = img[None, ...]
        gt = load_mhd(get_ground_truth(patient.file))
            
        for n, o in enumerate(img):
            patient['sim number'] = n
            patient['rmse'] = rmse(o, gt)
            rows_list.append(pd.DataFrame(patient).T)
    return pd.concat(rows_list, ignore_index=True)

def calculate_noise_reduction(results, measure='noise std'):
    cols = ['phantom', 'FOV [cm]', 'recon', 'Dose [%]']
    means = results[[*cols, measure]].groupby(cols).mean()
    noise_reductions = []
    for idx, row in results.iterrows():
        fbp_noise = means[measure][row.phantom, row['FOV [cm]'], 'fbp', row['Dose [%]']]
        noise_reductions.append(noise_reduction(fbp_noise, row[measure]))
    results[f'{measure} reduction [%]'] = noise_reductions
    return results

def calculate_task_improvement(results, measure='auc'):
    means = df.groupby(['effective diameter [cm]', 'recon', 'contrast [HU]', 'Dose [%]', 'observer'])['auc'].mean()
    noise_reductions = []
    for idx, row in results[results['experiment'] == 'task performance'].iterrows():
        fbp_noise = means[row['effective diameter [cm]'], 'fbp',row['contrast [HU]'], row['Dose [%]'], row['observer']]
        noise_reductions.append(row[measure] - fbp_noise)
    results.loc[results['experiment'] == 'task performance', f'delta {measure}'] = noise_reductions
    return results

def age_to_eff_diameter(age):
    # https://www.aapm.org/pubs/reports/rpt_204.pdf
    x = age
    a = 18.788598
    b = 0.19486455
    c = -1.060056
    d = -7.6244784
    y = a + b*x**1.5 + c *x**0.5 + d*np.exp(-x)
    eff_diam = y
    return eff_diam

adult_waist_circumferences_cm = {
    # 20: 90.7,
    30: 99.9,
    40: 102.8,
    # 50: 103.3,
    60: 106.2,
    70: 106.6,
    80: 104.1
}

def diameter_range_from_subgroup(subgroup):
    if subgroup == 'newborn': return (0, age_to_eff_diameter(1/12))
    elif subgroup == 'infant': return (age_to_eff_diameter(1/12), age_to_eff_diameter(2))
    elif subgroup == 'child': return (age_to_eff_diameter(2), age_to_eff_diameter(12))
    elif subgroup == 'adolescent': return (age_to_eff_diameter(12), age_to_eff_diameter(22))
    else: return (age_to_eff_diameter(22), 100)

def pediatric_subgroup(diameter):
    if diameter < age_to_eff_diameter(1):
        return 'newborn'
    elif (diameter >= age_to_eff_diameter(1)) & (diameter < age_to_eff_diameter(5)):
        return 'infant'
    elif (diameter >= age_to_eff_diameter(5)) & (diameter < age_to_eff_diameter(12)):
        return 'child'
    elif (diameter >= age_to_eff_diameter(12)) & (diameter < age_to_eff_diameter(22)):
        return 'adolescent'
    else:
        return 'adult'

def load_dicom(dcm_file):
    dcm = pydicom.dcmread(dcm_file)
    return (dcm.pixel_array + int(dcm.RescaleIntercept)).astype(float)
    
def convert_dicom_to_metaheader(metadata, phantom='ACR464'):
    names = []
    diameters = []
    fovs = []
    doses = []
    ctdivol = []
    recons = []
    kernels = []
    phantoms = []
    slices = []
    repeats = []
    files = []

    for name in metadata['Name'].unique():
        scan_df = metadata[metadata['Name'] == name]
        for diameter in scan_df['effective diameter (cm)'].unique():
            for dose in scan_df['Dose [%]'].unique():
                for fov in scan_df['FOV (cm)'].unique():
                    for kernel in scan_df['kernel'].unique():
                        for recon in scan_df['recon'].unique():
                            output_dir = base_dir / phantom / f'diameter{int(diameter*10):d}mm' / f'fov{int(fov*10)}mm' / f'dose_{dose:03d}' / kernel / recon

                            repeat = 0 if (name.startswith('ACR464 High') | name.startswith('pediatric_chest_phantom High')) else int(name.split('  ')[0].split('MAs')[1].split(' ')[0]) - 1
                            patient = scan_df[(scan_df['Dose [%]']==dose) &
                                               (scan_df['phantom'] == phantom) &
                                               (scan_df['FOV (cm)'] == fov) &
                                               (scan_df['recon'] == recon) &
                                               (scan_df['kernel'] == kernel) &
                                               (scan_df['phantom'] == 'ACR464') &
                                               (scan_df['effective diameter (cm)'] == diameter) &
                                               (scan_df['Name'] == name) &
                                               ((scan_df['slice'] > 159) & (scan_df['slice'] < 186))]
                            if len(patient) == 0: continue
                            assert(len(patient.Name.unique())== 1)
                            assert(len(patient) == 186-159-1)
                            output_dir.mkdir(exist_ok=True, parents=True)
                            vol = np.array([load_dicom(data_dir / dcm_file) for dcm_file in patient.file])
                            img = sitk.GetImageFromArray(vol)
                            fname = output_dir / f'{name}.mhd'
                            if repeat == 1: print(f'saving to: {fname}')
                            sitk.WriteImage(img, fname)

                            names.append(name)
                            diameters.append(diameter)
                            fovs.append(fov)
                            doses.append(dose)
                            recons.append(recon)
                            kernels.append(kernel)
                            phantoms.append(phantom)
                            repeats.append(repeat)
                            files.append(fname.relative_to(base_dir / phantom))

    preprocessed_metadata = pd.DataFrame({'Name': names,
                                          'effective diameter (cm)': diameters, 
                                          'FOV (cm)': fovs,
                                          'Dose [%]': doses,
                                          'recon': recons,
                                          'kernel': kernels,
                                          'phantom': phantoms,
                                          'repeat': repeats, 
                                          'file': files})
    return preprocessed_metadata

def concatenate_files_to_volumes(preprocessed_metadata):
    for fov in preprocessed_metadata['FOV (cm)'].unique():
        for dose in preprocessed_metadata['Dose [%]'].unique():
            for kernel in preprocessed_metadata['kernel'].unique():
                for recon in preprocessed_metadata['recon'].unique():
                    patient = preprocessed_metadata[(preprocessed_metadata['FOV (cm)'] == fov) &
                                                    (preprocessed_metadata['Dose [%]'] == dose) &
                                                    (preprocessed_metadata['recon'] == recon) &
                                                    (preprocessed_metadata['kernel'] == kernel)]
                    vol = np.concatenate([load_mhd(base_dir / phantom / f) for f in patient.file], axis=0)
                    img = sitk.GetImageFromArray(vol)
                    [os.remove(base_dir / phantom / f) for f in patient.file]
                    [os.remove(base_dir / phantom / f.parent / f'{f.stem}.raw') for f in patient.file]
                    fname = base_dir / phantom / patient.file.iloc[0]
                    print(fname.relative_to(base_dir))
                    sitk.WriteImage(img, fname)
    preprocessed_metadata = preprocessed_metadata[preprocessed_metadata.repeat==0].copy()
    preprocessed_metadata.pop('repeat')
    return preprocessed_metadata

# https://radiopaedia.org/articles/windowing-ct?lang=us
display_settings = {
    'brain': (80, 40),
    'subdural': (300, 100),
    'stroke': (40, 40),
    'temporal bones': (2800, 600),
    'soft tissues': (400, 50),
    'lung': (1500, -600),
    'liver': (150, 30),
}

def browse_studies(metadata, phantom='ACR464', fov=25, dose=100, recon='fbp', kernel='Qr43', repeat=0, display='soft tissues', slice_idx=0):
    patient = metadata[(metadata['Dose [%]']==dose) &
                       (metadata['phantom'] == phantom) &
                       (metadata['FOV (cm)']==fov) &
                       (metadata['recon'] == recon) &
                       (metadata['kernel'] == kernel) &
                       (metadata['repeat']==repeat) &
                       (metadata['slice']==slice_idx)]
    dcm_file = patient.file.item()
    dcm = pydicom.dcmread(dcm_file)
    img = dcm.pixel_array + int(dcm.RescaleIntercept)
    
    ww, wl = display_settings[display]
    minn = wl - ww/2
    maxx = wl + ww/2
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=minn, vmax=maxx)
    plt.colorbar(label=f'HU | {display} [ww: {ww}, wl: {wl}]')
    plt.title(patient['Name'].item())

from ipywidgets import interact, IntSlider

def study_viewer(metadata): 
    viewer = lambda **kwargs: browse_studies(metadata, **kwargs)
    slices = metadata['slice'].unique()
    interact(viewer,
             phantom=metadata.phantom.unique(),
             dose=sorted(metadata['Dose [%]'].unique(), reverse=True),
             fov=sorted(metadata['FOV (cm)'].unique()),
             recon=metadata['recon'].unique(),
             kernel=metadata['kernel'].unique(),
             repeat=metadata['repeat'].unique(),
             display=display_settings.keys(),
             slice_idx=IntSlider(value=slices[len(slices)//2], min=min(slices), max=max(slices)))
    
def make_metadata(data_dir):
    names = []
    diameters = []
    fovs = []
    doses = []
    ctdivol = []
    recons = []
    kernels = []
    phantoms = []
    slices = []
    repeats = []
    files = []

    dcm_files = sorted(list(data_dir.rglob('*.dcm')))

    for dcm_file in dcm_files:
        rel_path = dcm_file.relative_to(data_dir)
        recon, phantom, dose_kernel_fov, fname = rel_path.parts
        names.append(phantom + ' ' + dose_kernel_fov.replace('  1.0  ', '_').replace('_', ' ') + f' {recon}')
        diameter = 20.0 if (phantom == 'ACR464') else 27.2
        diameters.append(diameter)
        fovs.append(float(dose_kernel_fov.split('_')[1].split('mm')[0]) / 10)

        if dose_kernel_fov.startswith('High'):
            dose = 200
        elif dose_kernel_fov.startswith('Full'):
            dose = 100
        elif dose_kernel_fov.startswith('Quarter'):
            dose = 25
        doses.append(dose)
        recons.append(recon)
        kernels.append(dose_kernel_fov.split('1.0  ')[1].split('_')[0])
        phantoms.append(phantom)
        repeat = 0 if dose_kernel_fov.startswith('High') else int(dose_kernel_fov.split('  ')[0].split('MAs')[1]) - 1
        repeats.append(repeat)
        files.append(rel_path)

    metadata = pd.DataFrame({'Name': names,
                             'effective diameter (cm)': diameters, 
                             'FOV (cm)': fovs,
                             'Dose [%]': doses,
                             'recon': recons,
                             'kernel': kernels,
                             'phantom': phantoms,
                             'repeat': repeats, 
                             'file': files})
    slices = []
    for name in metadata['Name'].unique():
        slices += list(range(len(metadata[(metadata['Name']==name)])))
    metadata['slice'] = slices
    metadata['simulated'] = False
    return metadata

def estimate_ground_truth(preprocessed_metadata):
    for fov in preprocessed_metadata['FOV (cm)'].unique():
        patient = preprocessed_metadata[(preprocessed_metadata['kernel'] == 'Qr43') &
                                        (preprocessed_metadata['Dose [%]'] == 200) &
                                        (preprocessed_metadata['recon'] == 'fbp') &
                                        (preprocessed_metadata['FOV (cm)'] == fov)]
        fname = patient.file.item()
        vol = load_mhd(base_dir / phantom / fname)
        gt = vol.mean(axis=0)
        img = sitk.GetImageFromArray(gt)
        outfile = base_dir / phantom/ fname.parents[3] / 'true.mhd'
        print(outfile)
        sitk.WriteImage(img, outfile)

def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

def normalize(image): return 1 - (image.max() - image) / (image.max() - image.min())