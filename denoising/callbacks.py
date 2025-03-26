from pathlib import Path
import os

import numpy as np
import pydicom
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import Callback
from data import MayoLDGCDataModule, PediatricIQDataModule, HeadSimCTDataModule
from dotenv import load_dotenv
import wandb

load_dotenv()

def convert_to_dicom(img_slice: torch.tensor, phantom_path: str,
                     spacings:tuple|None=None):
    '''
    :param img_slice: input 2D ndarray to be saved
    :param phantom_path: filename to save dicom file to
    :param spacings: tuple containing pixel spacings in mm
    '''
    # https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144
    Path(phantom_path).parent.mkdir(exist_ok=True, parents=True)
    fpath = pydicom.data.get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(fpath)
    img_slice = img_slice.squeeze()
    ds.Rows, ds.Columns = img_slice.shape
    if spacings:
        ds.SliceThickness = spacings[0]
        ds.PixelSpacing = [spacings[1], spacings[2]]
    ds.PixelData = img_slice.copy(order='C').astype('int16') -\
        int(ds.RescaleIntercept)
    pydicom.dcmwrite(phantom_path, ds)


class DicomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for batch_idx, batch in zip(batch_indices[0], predictions):
            for idx, prediction in zip(batch_idx, batch):
                convert_to_dicom(np.array(prediction), self.output_dir / f"{idx:04d}.dcm")


class ImagePredictionLogger(Callback):
    def __init__(self, num_samples=5):
        super().__init__()
        dm = MayoLDGCDataModule(os.environ['LDGC_PATH'], region='chest', patch_size=None, batch_size=num_samples)
        dm.setup('fit')
        val_samples = next(iter(dm.val_dataloader()))
        
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs
        self.val_labels = self.val_labels

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        preds = pl_module(val_imgs).to('cpu')

        trainer.logger.experiment.log({
            "examples": [wandb.Image(torch.cat((x, y, pred), dim=2), caption=f"Image, Label, Pred")
                            for x, pred, y in zip(self.val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })