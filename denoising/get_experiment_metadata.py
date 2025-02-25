import os
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from data import PediatricIQDataset
from dotenv import load_dotenv

load_dotenv()


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
    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    parser = ArgumentParser('get experiment metadata', usage='''
    This function receives a directory path as input and generates metadata
    about experimental results within the directory.''')
    parser.add_argument('experiment_dir', type=str, help='''experiment directory path''')
    parser.add_argument('--output', '-o', type=str, default='metadata.csv', help='output file name csv')
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    metadata = get_experiment_metadata(experiment_dir)
    fname = experiment_dir / args.output
    print(f'{fname}')
    metadata.to_csv(fname, index=False)
