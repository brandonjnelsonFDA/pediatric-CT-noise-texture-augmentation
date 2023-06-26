# %%
from pathlib import Path
import urllib
import zipfile
import tensorflow as tf
import SimpleITK as sitk

data_dir = Path('data/CCT189')
if not data_dir.exists():
    data_dir.mkdir(parents=True)
    url = 'https://zenodo.org/record/7996580/files/large_dataset.zip?download=1'
    fname = str(data_dir / 'CCT189.zip')
    urllib.request.urlretrieve(url, fname)

    with zipfile.ZipFile(fname,"r") as zip_ref:
        zip_ref.extractall(fname.split('.zip')[0])

# %%
simple_cnn_denoiser = tf.keras.models.load_model('models/simple_cnn_denoiser')
simple_cnn_denoiser.summary()
# %%
model_vggloss = tf.keras.models.load_model('models/model_vggloss', compile=False)
model_vggloss.summary()
# %%
import matplotlib.pyplot as plt
import SimpleITK as sitk
# %%
from pathlib import Path
# %%
def denoise(input_dir, output_dir=None, model=None, name=None, offset=1000, batch_size=10):
    for series in input_dir.rglob('*.mhd'):
        if series.stem == 'ground_truth':
            continue
        input_image = sitk.ReadImage(series)
        x, y, z = input_image.GetWidth(), input_image.GetHeight(), input_image.GetDepth()
        input_array = sitk.GetArrayViewFromImage(input_image).reshape(z, x, y, 1).astype('float32') - offset
        sp_denoised = model.predict(input_array, batch_size=batch_size)
        output = Path(str(series).replace(str(input_dir), str(output_dir)))
        output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(sp_denoised), output)
        print(f'{name} --> {output}')

model = simple_cnn_denoiser
datasets = [ 
            {
            'input_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp',
            'output_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp_denoised_mse',
            'model': simple_cnn_denoiser,
            'name': 'CCT189 simple CNN'
            }, 
            {
            'input_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp',
            'output_dir': data_dir / 'CCT189' / 'large_dataset' / 'fbp_denoised_vgg',
            'model': model_vggloss,
            'name': 'CCT189 simple CNN VGG Loss'
            },
            {
            'input_dir': data_dir / 'CCT189_peds',
            'output_dir': data_dir / 'CCT189_peds_denoised_mse',
            'model': simple_cnn_denoiser,
            'name': 'CCT189 ped sized simple CNN',
            },
            {
            'input_dir': data_dir / 'CCT189_peds',
            'output_dir': data_dir / 'CCT189_peds_denoised_vgg',
            'model': model_vggloss,
            'name': 'CCT189 ped sized simple CNN VGG Loss'
            }
            ]
for dataset in datasets:
    denoise(**dataset)