import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path

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
    gt_file = 'noise_free.mhd' if fname.stem.startswith('signal') else 'true.mhd'
    return Path(fname).parents[2] / gt_file

def center_crop(img):
    diam = 2*np.sqrt(np.sum(img > img.mean())/np.pi)
    buffer = int((img.shape[0]-diam)//2)
    return img[buffer:img.shape[0]-buffer, :]

def center_crop_like(other, ref):
    diam = 2*np.sqrt(np.sum(ref > ref.mean())/np.pi)
    buffer = int((ref.shape[0]-diam)//2)
    return other[buffer:ref.shape[0]-buffer, :]

def make_montage(meta_df:pd.DataFrame, dose:int=25, diameters:list=[35.0, 11.2], recons:list = ['fbp', 'RED-CNN', 'RED-CNN augmented'],
                 phantom:str = 'MITA-LCD', roi_diameter:float|int=0.4, roi_center:tuple|str=(256, 256),  wwwl = (80, 0), crop_to_fit=True):
    """
    make image montage based on given argument parameters. Recons are plotted horizontally along the x axis while different diameters are plotted on y
    :Parameters:
        :meta_df: metadata dataframe
        :dose: dose level in percent [%]
        :diameters: phantom effective diameter in cm
        :recons: list of recon kernels to display on the horizontal x-axis
        :phantom: which image phantom is expected ['MITA-LCD', 'uniform', 'anthropomorphic'], it should be in meta_df
        :roi_diameter: diameter of circlular ROI. If `roi_diameter` is an **integer** then roi_diameter is in pixels e.g. 100, else if `roi_diameter` is **float** e.g. 0.3 then it is assumed a fraction of the phantom's effective diameter. Note for noise measurements IEC standard suggests centred circle ROI 40% of phantom diameter
        :roi_center: xy coordinates for roi center, or organ e.g. liver, if phantom = 'anthropomorphic'. If `str` is provided an is in the organ list `['liver']` a **random** centered roi that fits in the organ will be provided. If `roi_center` is a tuple, e.g. (256, 256) then an roi at those exact coordinates will be given
        :wwwl: window width and window level display settings, examples: soft tissues (400, 50), liver (150, 30) for recommend display settings see <https://radiopaedia.org/articles/windowing-ct?lang=us> for more information
    """
    
     #relative to phantom diameter (decrease below the recommended 40% diameter to fit between the inserts
    # assert(phantom in ['MITA-LCD', 'uniform'])    
    all_imgs = []
    all_gts = []
    idx = 0
    for diameter in diameters:
        recon_imgs = []
        recon_gts = []
        available_diameters = sorted(meta_df['effective diameter [cm]'].unique())
        if diameter not in available_diameters: raise ValueError(f'diameter {diameter} not in {available_diameters}')
        for recon in recons:
            offset = 1000 if recon == 'fbp' else 0
            filt = (meta_df['effective diameter [cm]'] == diameter) & (meta_df['Dose [%]'] == dose) & (meta_df['phantom']==phantom)
            mhd_file = meta_df[(meta_df.recon == recon) & filt].file.item()
            img = load_mhd(mhd_file).squeeze()[idx] - offset
            gt = load_mhd(get_ground_truth(mhd_file))-1000
            if crop_to_fit:
                gt = center_crop_like(gt, img)
                img = center_crop(img)
            recon_imgs.append(img)
            recon_gts.append(gt)
        all_imgs.append(recon_imgs)
        all_gts.append(recon_gts)

    if phantom in ['MITA-LCD', 'uniform']:  
        phantom_diameter_px = get_circle_diameter(all_imgs[0][0])
    else:
        phantom_diameter_px = all_imgs[0][0].shape[1]/1.1

    circle_selection_diameter_px = roi_diameter if isinstance(roi_diameter, int) else roi_diameter*phantom_diameter_px

    if isinstance(roi_center, tuple|list):
        circle_selections = np.array([len(all_imgs[0])*[circle_select(all_imgs[0][0], roi_center, r = circle_selection_diameter_px/2)] for _ in all_imgs])
    elif isinstance(roi_center, str):
        if phantom not in ['anthropomorphic']: raise ValueError(f'str roi center {roi_center} not available for phantom type {phantom}, consider setting `roi_center` to a tuple  e.g. (256, 256)')
        available_organs = {'liver': (50, 100)}
        if roi_center not in available_organs: raise ValueError(f'roi center {roi_center} not in {available_organs}')
        circle_selections = [len(all_imgs[0])*[add_random_circle_lesion(img[0], mask=((gt[0] >= 50) & (gt[0] < 100)), radius=circle_selection_diameter_px/2)[1].astype(bool)] for img, gt in zip(all_imgs, all_gts)]
        
    immatrix = np.concatenate([np.concatenate(row, axis=1) for row in all_imgs], axis=0)
    ctshow(immatrix, wwwl)
    plt.colorbar(fraction=0.015, pad=0.01, label='HU')
    immatrix = np.concatenate([np.concatenate(row, axis=1) for row in circle_selections], axis=0)
    plt.imshow(immatrix, alpha=0.1, cmap='Reds')
    for didx, diam in enumerate(all_imgs):
        for ridx, recon in enumerate(diam):
            nx, ny = recon.shape
            plt.annotate(f'mean: {recon[circle_selections[didx][ridx]].mean():2.0f} HU\nstd: {recon[circle_selections[didx][ridx]].std():2.0f} HU',
                         (ny//2 + ny*ridx, nx//2 + nx*didx), fontsize=6, bbox=dict(boxstyle='square,pad=0.3', fc="lightblue", ec="steelblue"))
    plt.title(' | '.join(recons))
    plt.ylabel(' cm |'.join(map(lambda o: str(o), diameters[::-1])) + ' mm')
    
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
    default_diameters = {'uniform': 0.4, 'MITA-LCD':0.3, 'anthropomorphic': 0.2}
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
        offset = 1000 if patient.recon == 'fbp' else 0
        img = load_mhd(patient.file) - offset
        if img.ndim == 2: img = img[None, ...]
        gt = load_mhd(get_ground_truth(patient.file)) - 1000
        
        if patient.phantom in ['uniform', 'MITA-LCD']:
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
        offset = 1000 if patient.recon == 'fbp' else 0
        img = load_mhd(patient.file) - offset
        if img.ndim == 2: img = img[None, ...]
        gt = load_mhd(get_ground_truth(patient.file)) - 1000
            
        for n, o in enumerate(img):
            patient['sim number'] = n
            patient['rmse'] = rmse(o, gt)
            rows_list.append(pd.DataFrame(patient).T)
    return pd.concat(rows_list, ignore_index=True)

def calculate_noise_reduction(results, measure='noise std'):
    cols = ['effective diameter (cm)', 'recon', 'Dose [%]']
    means = results[[*cols, measure]].groupby(cols).mean()
    noise_reductions = []
    for idx, row in results.iterrows():
        fbp_noise = means[measure][row['effective diameter (cm)'], 'fbp', row['Dose [%]']]
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