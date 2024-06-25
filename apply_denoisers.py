# apply_denoisers.py
# %%
from pathlib import Path
import urllib
import zipfile
from argparse import ArgumentParser

import SimpleITK as sitk
import pydicom

from denoising.networks import RED_CNN

import os
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

def denoise(input_dir, output_dir=None, kernel='fbp', model=None, name=None, offset=0, batch_size=32, overwrite=True, extensions = ['.mhd', '.dcm']):

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_dir = output_dir or input_dir
    if isinstance(extensions, str): extensions = [extensions]
    series_list = []
    for ext in extensions:
        series_list += list(input_dir.rglob('*' + ext))

    for series in series_list:
        if kernel not in series.parts: continue
        if (series.stem == 'ground_truth') or (series.stem == 'noise_free') or (series.stem == 'true'):
            continue
        output = Path(str(series).replace(str(input_dir), str(output_dir)))
        if output.exists() & (not overwrite):
            print(f'{output} already found, skipping {name}')
        else:
            output = Path(str(output).replace('fbp', name))
            output.parent.mkdir(parents=True, exist_ok=True)

            if series.suffix == '.dcm':
                dcm_image = pydicom.dcmread(series)
                x, y, z = dcm_image.Columns, dcm_image.Rows, 1
                input_array = dcm_image.pixel_array + int(dcm_image.RescaleIntercept)
                input_array = input_array.reshape(z, 1, x, y).astype('float32')
            elif series.suffix == '.mhd':
                input_image = sitk.ReadImage(series)
                x, y, z = input_image.GetWidth(), input_image.GetHeight(), input_image.GetDepth()
                input_array = sitk.GetArrayViewFromImage(input_image).reshape(z, 1, x, y).astype('float32') - offset
            if batch_size > z:
                batch_size = z
            print(f'denoising {series} of {z} images in batches of {batch_size}')

            model.to(dev)
            denoised = model.predict(input_array, batch_size=batch_size, device=dev)

            if series.suffix == '.mhd':
                output_image = sitk.GetImageFromArray(denoised.squeeze())
                assert((output_image.GetDepth(),output_image.GetHeight(),output_image.GetWidth())==
                        (input_image.GetDepth(), input_image.GetHeight(), input_image.GetWidth()))
                output_image.SetSpacing(input_image.GetSpacing())
                sitk.WriteImage(output_image, output)
                check_output_image = sitk.ReadImage(output)
                assert((check_output_image.GetDepth(),check_output_image.GetHeight(), check_output_image.GetWidth())==
                        (input_image.GetDepth(), input_image.GetHeight(), input_image.GetWidth()))
            if series.suffix == '.dcm':
                dcm_image.ConvolutionKernel += f' {name}'
                dcm_image.PixelData = (denoised - int(dcm_image.RescaleIntercept)).astype('uint16')
                pydicom.write_file(output, dcm_image)
            print(f'{name} --> {output}')
# %%

if __name__ == '__main__':
    parser = ArgumentParser(description='Apply denoiser')
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
