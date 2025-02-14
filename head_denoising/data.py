import os
from pathlib import Path

import pandas as pd
import pydicom
import torch
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

import lightning as L
from torch.utils.data import DataLoader, random_split


def read_image(path):
    dcm = pydicom.dcmread(path)
    return dcm.pixel_array + float(dcm.RescaleIntercept)

def get_patch(img, target, patch_size):
    patched_img = img.unfold(0, patch_size, patch_size//2).unfold(1, patch_size, patch_size//2)
    patched_target = target.unfold(0, patch_size, patch_size//2).unfold(1, patch_size, patch_size//2)
    ix, iy = torch.randint(0, patched_img.shape[0]-1, size=(2, 1))
    return patched_img[ix, iy], patched_target[ix, iy]

class LDHeadCTDataset(VisionDataset):
    '''
    Low Dose Head CT Dataset

    images : low dose head CT images
    target : routine dose head CT images of the same patient
    '''
    def __init__(self,
                 root=os.getcwd(),
                 train: bool=True,
                 transform=None,
                 target_transform=None,
                 download=True,
                 patch_size=None,
                 testid='case_000'):

      base_dir = Path(root)
      if download & (not base_dir.exists()):
        utils.download_and_extract_archive(url='<url not specified yet>',
                                           download_root=root)
      # build metadata file
      rd_metadata = pd.concat([pd.read_csv(o) for o in sorted(list((base_dir / 'rd').rglob('metadata_*.csv')))], ignore_index=True)
      ld_metadata = pd.concat([pd.read_csv(o) for o in sorted(list((base_dir / 'ld').rglob('metadata_*.csv')))], ignore_index=True)
      metadata = pd.concat([ld_metadata, rd_metadata], ignore_index=True)

      # assign slice labels
      for case in metadata['name'].unique():
        for mA in metadata['mA'].unique():
            metadata.loc[(metadata['name']==case) & 
                         (metadata['mA'] == mA), 'slice'] = list(range(len(metadata[(metadata['name']==case) &
                                                                                    (metadata['mA'] == mA)])))

      if train:
        metadata = metadata[metadata['name'] != testid]
      else:
        metadata = metadata[metadata['name'] == testid]
      self.root = base_dir
      self.metadata = metadata
      self.patch_size = patch_size
      self.ld_metadata = self.metadata[self.metadata['mA'] == 60]
      self.rd_metadata = self.metadata[self.metadata['mA'] == 240]
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):
      return len(self.ld_metadata)

    def __getitem__(self, idx):
      ld_patient = self.ld_metadata.iloc[idx]
      ld_img_path = self.root / ld_patient['image file']
      image = read_image(ld_img_path)

      rd_patient = self.rd_metadata[(self.rd_metadata['name'] == ld_patient['name']) &
                                    (self.rd_metadata['slice'] == ld_patient['slice'])]
      rd_img_path = self.root / rd_patient['image file'].item()
      label = read_image(rd_img_path)
      if self.transform:
        image = self.transform(image)
      if self.target_transform:
        label = self.target_transform(label)

      if self.patch_size:
        image, label = get_patch(image.squeeze(),
                                 label.squeeze(),
                                  self.patch_size)
      return image, label


class LDHeadCTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", patch_size=64, batch_size=32, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

    def prepare_data(self):
        # download
        # LDHeadCTDataset(base_dir)
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_set = LDHeadCTDataset(self.data_dir, train=True, patch_size=self.patch_size,
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
            self.test_set = LDHeadCTDataset(self.data_dir, train=False,
                                transform=self.transform, target_transform=self.transform)

        if stage == "predict":
            self.predict_set = LDHeadCTDataset(self.data_dir, train=False,
                                transform=self.transform, target_transform=self.transform)

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
    '''
    For Mayo Clinic Low Dose CT Dataset, note this dataset
    includes, head, chest, and abdomen, need to be careful which I select
    '''
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
