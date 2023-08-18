# %%
from pathlib import Path
import urllib
import zipfile
from argparse import ArgumentParser

import tensorflow as tf
import SimpleITK as sitk


simple_cnn_denoiser = tf.keras.models.load_model('models/simple_cnn_denoiser')
simple_cnn_denoiser_augmented = tf.keras.models.load_model('models/simple_cnn_denoiser_augmented')
model_vggloss = tf.keras.models.load_model('models/model_vggloss', compile=False) #compile = False means don't need to load custom loss function


def denoise(input_dir, output_dir=None, model=None, name=None, offset=1000, batch_size=10, overwrite=True):
    for series in input_dir.rglob('*.mhd'):
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
            input_array = sitk.GetArrayViewFromImage(input_image).reshape(z, x, y, 1).astype('float32') - offset
            sp_denoised = model.predict(input_array, batch_size=batch_size)

            sitk.WriteImage(sitk.GetImageFromArray(sp_denoised), output)
            print(f'{name} --> {output}')

if __name__ == '__main__':

    parser = ArgumentParser(description='Runs XCIST CT simulations on XCAT datasets')
    parser.add_argument('base_directory', type=str, default="", help='directory containing images to be processed')
    args = parser.parse_args()

    data_dir = args.base_directory or 'data'
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        url = 'https://zenodo.org/record/7996580/files/large_dataset.zip?download=1'
        fname = str(data_dir / 'CCT189.zip')
        urllib.request.urlretrieve(url, fname)

        with zipfile.ZipFile(fname,"r") as zip_ref:
            zip_ref.extractall(fname.split('.zip')[0])
    datasets = [ 
                # {
                # 'input_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp',
                # 'output_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp_denoised_mse',
                # 'model': simple_cnn_denoiser,
                # 'name': 'simple CNN'
                # }, 
                # {
                # 'input_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp',
                # 'output_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp_denoised_vgg',
                # 'model': model_vggloss,
                # 'name': 'simple CNN VGG Loss'
                # },
                {
                'input_dir': data_dir / 'CCT189_peds_fbp',
                'output_dir': data_dir / 'CCT189_peds_denoised_mse',
                'model': simple_cnn_denoiser,
                'name': 'simple CNN MSE',
                },
                {
                'input_dir': data_dir / 'CCT189_peds_fbp',
                'output_dir': data_dir / 'CCT189_peds_denoised_mse_w_augmentation',
                'model': simple_cnn_denoiser_augmented,
                'name': 'simple CNN MSE with augmentation',
                },
                # {
                # 'input_dir': data_dir / 'CCT189_peds_fbp',
                # 'output_dir': data_dir / 'CCT189_peds_denoised_vgg',
                # 'model': model_vggloss,
                # 'name': 'simple CNN VGG Loss'
                # }
                ]
    for dataset in datasets:
        denoise(**dataset)