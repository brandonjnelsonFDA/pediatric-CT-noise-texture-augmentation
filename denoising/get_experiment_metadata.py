import os
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from data import PediatricIQDataset
from dotenv import load_dotenv

load_dotenv()

def get_repeat(recon_name):
    '''
    get the model iteration in the case of repeat trainings from the filename
    '''
    splits = recon_name.split('_')
    if len(splits) > 1:
        strength = int(splits[-1])
        return strength
    return 0


def get_strength(recon_name):
    '''
    get the augmentation strength float within the range [0, 1] from the filename
    '''
    splits = recon_name.split('augmented_')
    if len(splits) > 1:
        strength = float(splits[-1].split('_')[0])
        return strength
    return 0


def get_experiment_metadata(experiment_dir: str) -> pd.DataFrame:
    """
    This function receives a directory path as input and generates metadata about experimental results within the directory.

    Parameters:
    experiment_dir (str): The path to the directory containing experiment outputs

    Returns:
    pd.DataFrame: A DataFrame containing the metadata of experiments
    """
    experiment_dir = Path(experiment_dir)

    dfs = []
    dset = PediatricIQDataset(os.environ['PEDIATRICIQ_PATH'], train='predict')
    ld_metadata = dset.ld_metadata.copy()
    dfs.append(ld_metadata.copy())

    for test_set in ['PedIQ']:
        for recon_dir in experiment_dir.rglob(test_set):
            denoised_fnames = sorted(list(recon_dir.rglob('*.dcm')))        
            ld_metadata = dset.ld_metadata.copy()
            ld_metadata['file'] = denoised_fnames
            ld_metadata['recon'] = ld_metadata['file'].apply(lambda o: o.parts[-3])
            dfs.append(ld_metadata)
    metadata = pd.concat(dfs, ignore_index=True)
    metadata['lambda'] = metadata['recon'].apply(get_strength)
    metadata['repeat'] = metadata['recon'].apply(get_repeat)
    return metadata


if __name__ == '__main__':
    parser = ArgumentParser(
        description='This script generates metadata about experimental results within a specified directory.',
        usage='python %(prog)s experiment_dir [-o output_file]')
    parser.add_argument(
        'experiment_dir',
        type=str,
        help='Path to the directory containing the experimental output files.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='metadata.csv',
        help='Name of the output CSV file. Default is "metadata.csv". '
             'If another filename is provided, it should include the .csv extension.'
    )
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    metadata = get_experiment_metadata(experiment_dir)
    fname = experiment_dir / args.output
    print(f'{fname}')
    metadata.to_csv(fname, index=False)
