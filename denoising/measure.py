import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from data import read_image


def circle_select(img, xy, r):
    """
    Create a boolean mask for a circle with the given center and radius on an image.

    Parameters:
    img (numpy.ndarray): The 2D image on which to draw the circle.

    xy (tuple): The center coordinates of the circle.

    r (int): The radius of the circle.

    Returns:
    numpy.ndarray: A boolean mask with the same shape as the input image, where the elements inside the circle are True.
    """
    assert img.ndim == 2
    circle_mask = np.zeros_like(img)
    for i in range(circle_mask.shape[0]):
        for j in range(circle_mask.shape[1]):
            if (i-xy[0])**2 + (j-xy[1])**2 < r**2:
                circle_mask[i, j] = True
    return circle_mask.astype(bool)


def measure_image(fname, measure='mean', r=100, xy=(256, 256)):
    """
    Calculate a statistical measure on a circular region of interest (ROI) on an image.

    Parameters:
    fname (str): The filename of the image.

    measure (str, optional): The statistical measure to calculate.
                             Currently supported measures are 'mean', 'std', and 'bias'.
                             Default is 'mean'.

    r (int, optional): The radius of the circular ROI.
                       Default is 100.

    xy (tuple, optional): The center coordinates of the circular ROI.
                          Default is (256, 256).

    Returns:
    float: The computed statistical measure.
    """
    img = read_image(fname)
    roi = circle_select(img, r=r, xy=xy)
    func = eval('np.'+measure)
    return func(img[roi])


def measure_bias(uniform):
    """
    Calculate the bias of the 'mean' measure compared to the 'mean' measure of the 'fbp'
    reconstruction for each unique 'name' and 'recon' combination in the input DataFrame.

    Parameters:
    uniform (pd.DataFrame): A DataFrame with columns 'name', 'recon', and 'mean'.

    Returns:
    pd.DataFrame: The input DataFrame with a new 'bias' column.
    """
    for name in uniform.name.unique():
        for recon in uniform.recon.unique():
            fbp = uniform[(uniform.name==name) & (uniform.recon=='fbp')]['mean'].to_numpy()
            uniform.loc[(uniform.name==name) & (uniform.recon==recon), 'bias'] = uniform[(uniform.name==name) & (uniform.recon==recon)]['mean'].to_numpy() - fbp
    return uniform

def make_uniform_phantom_measures(metadata: pd.DataFrame, phantom='uniform', measure=['mean', 'std', 'bias'],
                                  xy=(256, 256), r=100) -> pd.DataFrame:
    """
    Generate various statistical measures from the uniform phantom's image files.

    Parameters:
    metadata (pd.DataFrame): A DataFrame consisting of columns 'phantom' and 'file'.
                             'phantom' contains the type of phantom and 'file' contains the file location of the image.

    phantom (str, optional): The specific type of phantom for which measures are to be calculated.
                             If specified, the function will only consider rows where the 'phantom' column value matches this value.
                             Default is 'uniform'.

    measure (list, optional): A list of statistical measures to calculate for each image.
                             Currently supported measures are 'mean', 'std', and 'bias'.
                             Default is ['mean', 'std', 'bias'].

    xy (tuple, optional): The center coordinates of the circular ROI on the image.
                          Default is (256, 256).

    r (int, optional): The radius of the circular ROI on the image.
                       Default is 100.

    Returns:
    pd.DataFrame: The input DataFrame with new columns for each statistical measure in the 'measure' list.
                  For 'bias' measure, a new 'bias' column will be added to the DataFrame.
    """
    if phantom:
        metadata = metadata[metadata['phantom'] == phantom]
    for measurand in measure:
        print(f'Now measuring: {measurand} on {len(metadata)} images')
        if measurand == 'bias':
            metadata = measure_bias(metadata)
        else:
            metadata[measurand] = metadata['file'].apply(lambda o: measure_image(o, measure=measurand, r=r, xy=xy))    
    return metadata


def main(metadata_fname, results_fname='uniform_roi_measures.csv', phantom='uniform', measure=['mean', 'std', 'bias'],
                                  xy=(256, 256), r=100):
    """
    Load metadata from a csv file, generate statistical measures for a specific type of phantom,
    and save the results to a csv file.

    Parameters:
    metadata_fname (str): The path to the csv file containing the metadata.
                          This file must include a 'phantom' column for the phantom type
                          and a 'file' column for the file location.
                          It should contain data for a uniform phantom.

    results_fname (str, optional): The filename of the csv file to save the results.
                                   The results will be saved in the current directory.
                                   Default is 'uniform_roi_measures.csv'.

    phantom (str, optional): The specific type of phantom for which measures are to be calculated.
                             If specified, the function will only consider rows where the 'phantom' column value matches this value.
                             Default is 'uniform'.

    measure (list, optional): A list of statistical measures to calculate for each image.
                             Currently supported measures are 'mean', 'std', and 'bias'.
                             Default is ['mean', 'std', 'bias'].

    xy (tuple, optional): The center coordinates of the circular ROI on the image.
                          Default is (256, 256).

    r (int, optional): The radius of the circular ROI on the image.
                       Default is 100.
    """
    uniform = pd.read_csv(metadata_fname)
    uniform = make_uniform_phantom_measures(uniform)
    print(f"Saving results to {results_fname}")
    uniform.to_csv(results_fname, index=False)

if __name__ == '__main__':
    parser = ArgumentParser(description='This script generates statistical measures for a uniform phantom from image files. It reads the metadata from a csv file, calculates the measures, and saves the results to a new csv file.',
                            usage='python %(prog)s metadata_file [-o output_file]')
    parser.add_argument('metadata_file', nargs='?', help='The path to the csv file containing the metadata. The file must include a column for phantom type and file location. It should contain data for a uniform phantom.')
    parser.add_argument('--output', '-o', metavar='OUTPUT_FILE', help='The name of the output file where the results will be saved. Default is "uniform_roi_measures.csv" in the metadata_file directory.')
    parser.add_argument('--phantom', type=str, default='uniform', help="The specific type of phantom for which measures are to be calculated. If specified, the function will only consider rows where the 'phantom' column value matches this value.")
    args = parser.parse_args()
    if args.metadata_file:
        metadata_file = args.metadata_file
    elif not sys.stdin.isatty():
        metadata_file = sys.stdin.read().strip()
    else:
        parser.print_help()
    output = args.output
    if not output:
        output = Path(metadata_file).parent / 'uniform_roi_measures.csv'
    
    main(metadata_fname=metadata_file, results_fname=output)
