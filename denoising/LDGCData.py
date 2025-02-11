import os
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import utils, VisionDataset
from torch.utils.data import DataLoader, random_split
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pydicom

def read_image(path):
    dcm = pydicom.dcmread(path)
    return dcm.pixel_array + int(dcm.RescaleIntercept)


def get_patch(img, target, patch_size):
    patched_img = img.unfold(0,
                             patch_size,
                             patch_size//2).unfold(1,
                                                   patch_size,
                                                   patch_size//2)
    patched_target = target.unfold(0,
                                   patch_size,
                                   patch_size//2).unfold(1,
                                                         patch_size,
                                                         patch_size//2)
    ix, iy = torch.randint(0, patched_img.shape[0]-1, size=(2, 1))
    return patched_img[ix, iy], patched_target[ix, iy]


def compute_patch_number(patch_size=64, img_size=512):
    if patch_size is None:
        return 1
    return np.prod(torch.ones(512, 512).unfold(0, patch_size, patch_size//2).unfold(1, patch_size, patch_size//2).shape[:2])


class MayoLDGCDataset(VisionDataset):
    def __init__(self,
                 root=os.getcwd(),
                 train: bool = True,
                 transform=None,
                 target_transform=None,
                 download=True,
                 patch_size: int | None = None,
                 testid=['L004']):

        root = Path(root)
        if download & (not root.exists()):
            utils.download_and_extract_archive(url='<url not provided>',
                                               download_root=root)
        data_dir = root / 'LDCT-and-Projection-data'
        metadata = pd.read_csv(root / 'metadata.csv')
        if train:
            metadata = metadata[~metadata['Subject ID'].isin(testid)]
        else:
            metadata = metadata[metadata['Subject ID'].isin(testid)]
        self.metadata = metadata
        self.root = root
        self.data_dir = data_dir

        self.image_paths = []
        self.target_paths = []

        for subject_id in self.metadata['Subject ID']:
            series = 'Low Dose Images'
            ld_study = metadata[(metadata['Subject ID'] == subject_id) &
                                (metadata['Series Description'] == series)]
            if len(ld_study) < 1:
                continue
            rd_study = metadata[(metadata['Subject ID'] == subject_id) &
                                (metadata['Series Description'] == series)]

            ld_path = self.root / ld_study['File Location'].item()[2:]
            rd_path = self.root / rd_study['File Location'].item()[2:]
            self.image_paths.extend(sorted(list(ld_path.glob('*.dcm'))))
            self.target_paths.extend(sorted(list(rd_path.glob('*.dcm'))))

        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ld_path = self.image_paths[idx]
        rd_path = self.target_paths[idx]

        image = read_image(ld_path)[None]
        label = read_image(rd_path)[None]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        if self.patch_size:
            image, label = get_patch(image.squeeze(),
                                     label.squeeze(),
                                     self.patch_size)
        return image, label


class MayoLDGCDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", patch_size=64, batch_size=32, num_workers=31):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

    def prepare_data(self):
        # download
        # MayoLDGCDataset()
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_set = MayoLDGCDataset(self.data_dir, train=True, patch_size=self.patch_size,
                                       transform=self.transform, target_transform=self.transform)
            # use 20% of training data for validation
            train_set_size = int(len(train_set) * 0.8)
            valid_set_size = len(train_set) - train_set_size

            # split the train set into two
            seed = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = random_split(train_set,
                                                        [train_set_size,
                                                         valid_set_size],
                                                        generator=seed)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = MayoLDGCDataset(self.data_dir, train=False,
                                            transform=self.transform,
                                            target_transform=self.transform)

        if stage == "predict":
            self.predict_set = MayoLDGCDataset(self.data_dir, train=False,
                                               transform=self.transform,
                                               target_transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size)
