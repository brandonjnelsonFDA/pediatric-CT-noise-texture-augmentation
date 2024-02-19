# %%
from denoising.loader import get_loader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

data_path = '/gpfs_projects/brandon.nelson/Mayo_LDGC/images'
saved_path = '/gpfs_projects/brandon.nelson/Mayo_LDGC/numpy_files'
test_patient='L506'
patch_n=10
patch_size=64

data_loader = get_loader(mode='train',
                         load_mode=0,
                         saved_path=saved_path,
                         test_patient=test_patient,
                         patch_n=patch_n,
                         patch_size=patch_size,
                         transform=False,
                         batch_size=16,
                         num_workers=7)
# %%

for iter_, (x, y) in enumerate(data_loader):
    if iter_ > 1:
        break
x.shape, y.shape
# %%
noise_patch_dir = Path('noise_patches/patch_size_64x64')
# diameters = [112, 131, 151, 185, 200, 216, 292, 350]
diameters = [112, 131, 151, 185, 216, 292]

noise_files = [noise_patch_dir / f'diameter{d}mm.npy' for d in diameters]
noise_patch_dict = {f.stem: np.load(f) for f in noise_files}
noise_patches = np.concatenate(list(noise_patch_dict.values()))
# %%
import matplotlib.pyplot as plt
def ctshow(im, vmin=None, vmax=None):
    plt.subplots(1,1, dpi=300)
    im = plt.imshow(im,cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.015, pad=0.01, label='HU')
    plt.axis('off')
# %%
ctshow(np.concatenate([np.concatenate([o for o in x[1, :3]], axis=0),
                       np.concatenate([o for o in y[1, :3]])], axis=1)
        )
plt.title('Input | Target')
# %%
