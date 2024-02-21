# %%
from pathlib import Path
import urllib
import zipfile
from argparse import ArgumentParser

# import tensorflow as tf
import SimpleITK as sitk

from denoising.networks import RED_CNN

import os
import sys
import torch
from collections import OrderedDict


def load_model(save_path, iter_=13000, multi_gpu=False):
    REDCNN = RED_CNN()
    f = os.path.join(save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
    if multi_gpu:
        state_d = OrderedDict()
        for k, v in torch.load(f):
            n = k[7:]
            state_d[n] = v
        REDCNN.load_state_dict(state_d)
        return REDCNN
    else:
        REDCNN.load_state_dict(torch.load(f))
        return REDCNN

cnn_denoiser = load_model('denoising/models/redcnn')
cnn_denoiser_augmented = load_model('denoising/models/redcnn_augmented')

# %%
def denoise(input_dir, output_dir=None, kernel='fbp', model=None, name=None, offset=1000, batch_size=32, overwrite=True):

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_dir = output_dir or input_dir
    for series in input_dir.rglob('*.mhd'):
        if kernel not in series.parts: continue
        if series.stem == 'ground_truth':
            continue
        input_image = sitk.ReadImage(series)
        output = Path(str(series).replace(str(input_dir), str(output_dir)))
        if output.exists() & (not overwrite):
            print(f'{output} already found, skipping {name}')
        else:
            output = Path(str(output).replace('fbp', name))
            output.parent.mkdir(parents=True, exist_ok=True)
            x, y, z = input_image.GetWidth(), input_image.GetHeight(), input_image.GetDepth()
            input_array = sitk.GetArrayViewFromImage(input_image).reshape(z, 1, x, y).astype('float32') - offset
            print(f'denoising {series} of {z} images in batches of {batch_size}')

            model.to(dev)
            sp_denoised = model.predict(input_array, batch_size=batch_size, device=dev)

            output_image = sitk.GetImageFromArray(sp_denoised.squeeze())
            assert((output_image.GetDepth(),output_image.GetHeight(),output_image.GetWidth())==
                    (input_image.GetDepth(), input_image.GetHeight(), input_image.GetWidth()))
            sitk.WriteImage(output_image, output)
            check_output_image = sitk.ReadImage(output)
            assert((check_output_image.GetDepth(),check_output_image.GetHeight(), check_output_image.GetWidth())==
                    (input_image.GetDepth(), input_image.GetHeight(), input_image.GetWidth()))
            print(f'{name} --> {output}')
# %%

if __name__ == '__main__':
    parser = ArgumentParser(description='Runs XCIST CT simulations on XCAT datasets')
    parser.add_argument('base_directory', type=str, default="data", help='directory containing images to be processed')
    parser.add_argument('--kernel', type=str, default="fbp", help='input kernel to be processed')
    args = parser.parse_args()

    data_dir = args.base_directory
    data_dir = Path(data_dir)
    kernel = args.kernel
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        url = 'https://zenodo.org/record/7996580/files/large_dataset.zip?download=1'
        fname = str(data_dir / 'CCT189.zip')
        urllib.request.urlretrieve(url, fname)

        with zipfile.ZipFile(fname,"r") as zip_ref:
            zip_ref.extractall(fname.split('.zip')[0])
    datasets = [ 
                {
                'input_dir': data_dir,
                'output_dir': data_dir,
                'kernel': kernel,
                'model': cnn_denoiser,
                'name': 'RED-CNN',
                },
                {
                'input_dir': data_dir,
                'output_dir': data_dir,
                'kernel': kernel,
                'model': cnn_denoiser_augmented,
                'name': 'RED-CNN augmented',
                },
                ]
    for dataset in datasets:
        denoise(**dataset)
# %%
