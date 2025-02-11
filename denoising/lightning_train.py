# %% imports
# %%
import sys

import matplotlib.pyplot as plt
from LDGCData import MayoLDGCDataModule, MayoLDGCDataset
import torch

from networks import RED_CNN
import torch.nn.functional as F
import lightning as L
# sys.path.append('..')
# from notebooks.utils import ctshow
torch.set_float32_matmul_precision('medium')

# %%

# %%
from torchvision.transforms import v2

tfms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
saved_path = '/gpfs_projects/common_data/DLIR/TCIA/manifest-1648648375084/'
train_set = MayoLDGCDataset(saved_path, train=True, patch_size=64,
                            transform=tfms, target_transform=tfms)
train_set
# %%
dm = MayoLDGCDataModule(saved_path, )
dm.prepare_data()
dm.setup('fit')
dl = dm.train_dataloader()
# %%
# %%timeit
for x, y in dl:
    print(x.shape, y.shape)
    break
# %%
idx = 30
f, axs = plt.subplots(1, 2, dpi=150)
axs[0].imshow(x[idx].squeeze(), cmap='gray')
axs[1].imshow(y[idx].squeeze(), cmap='gray')
# %%


class LitAutoEncoder(L.LightningModule):
    def __init__(self, torch_module):
        super().__init__()
        self.torch_module = torch_module

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.torch_module(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.torch_module(x)
        val_loss = F.mse_loss(x_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x_hat = self.torch_module(x)
        test_loss = F.mse_loss(x_hat, y)
        self.log("test_loss", test_loss)

    def forward(self, x):
        return self.torch_module(x)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir=lightning_logs/
# %%

dm = MayoLDGCDataModule(saved_path)
model = LitAutoEncoder(RED_CNN())
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)
# %%
trainer.test(datamodule=dm)
# %%
trainer.validate(datamodule=dm)
# %%
trainer.predict(datamodule=dm)

# %% test before fitting
pre_test = trainer.predict(datamodule=dm)
pre_test

idx = 6
f, axs = plt.subplots(1, 3, dpi=150)
trainer = L.Trainer()
res = trainer.predict(model, dm)
xhat = res[0].detach().numpy()[0]
axs[0].imshow(x.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
axs[1].imshow(xhat, cmap='gray', vmin=vmin, vmax=vmax)
axs[2].imshow(x.squeeze() - xhat, cmap='gray', vmin=vmin, vmax=vmax)
# %%
# 0.75 it/s with 1 worker