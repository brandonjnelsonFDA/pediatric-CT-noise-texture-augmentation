import os
from pathlib import Path

import pandas as pd
import pydicom
import torch
from torchvision.datasets import VisionDataset


def read_image(path):
  dcm = pydicom.dcmread(path)
  return dcm.pixel_array + int(dcm.RescaleIntercept)

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