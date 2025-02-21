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


class HeadSimCTDataset(VisionDataset):
    '''
    A PyTorch dataset for the head simulation CT dataset.

    This dataset includes low dose head CT images and their corresponding routine dose images.
    Args:
        root (str, optional): The root directory of the dataset. Defaults to the current working directory.
        train (bool, optional): Whether to use the training set. Defaults to True.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.
        download (bool, optional): Whether to download the dataset if it's not already present in the root directory. Defaults to True.
        patch_size (int, optional): The size of patches to extract from the images. Defaults to None (do not extract patches).
        testid (str, optional): The subject ID to use for testing. Defaults to 'case_000'.

    Attributes:
        root (Path): The root directory of the dataset.
        metadata (DataFrame): A pandas DataFrame containing metadata for the dataset.
        patch_size (int): The size of patches to extract from the images.
        ld_metadata (DataFrame): A pandas DataFrame containing metadata for the low dose images.
        rd_metadata (DataFrame): A pandas DataFrame containing metadata for the routine dose images.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable): A function/transform that takes in the target and returns a transformed version.
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
      '''
      Returns the number of samples in the dataset.

      Returns:
          int: The number of samples in the dataset.
      '''
      return len(self.ld_metadata)

    def __getitem__(self, idx):
      '''
      Returns the sample and target at the given index.

      Args:
          idx (int): The index of the sample.

      Returns:
          tuple: A tuple containing the sample and target.
      '''
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


class HeadSimCTDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the head simulation CT dataset.

    This module handles the data preprocessing, splitting into train/validation/test datasets,
    and provides dataloaders for each split.

    Args:
        data_dir (str, optional): The root directory of the dataset. Defaults to the current working directory.
        patch_size (int, optional): The size of patches to extract from the images. Defaults to 64.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 32.
        num_workers (int, optional): The number of workers for the dataloaders. Defaults to 1.

    Attributes:
        data_dir (str): The root directory of the dataset.
        patch_size (int): The size of patches to extract from the images.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for the dataloaders.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        train_set (HeadSimCTDataset): The training dataset.
        val_set (HeadSimCTDataset): The validation dataset.
        test_set (HeadSimCTDataset): The test dataset.
        predict_set (HeadSimCTDataset): The prediction dataset.
    """

    def __init__(self, data_dir: str = "./", patch_size=64, batch_size=32, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

    def prepare_data(self):
        # download
        # HeadSimCTDataset(base_dir)
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_set = HeadSimCTDataset(self.data_dir, train=True, patch_size=self.patch_size,
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
            self.test_set = HeadSimCTDataset(self.data_dir, train=False,
                                transform=self.transform, target_transform=self.transform)

        if stage == "predict":
            self.predict_set = HeadSimCTDataset(self.data_dir, train=False,
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
    '''
    A PyTorch dataset for the Mayo Clinic Low Dose CT dataset.

    This dataset includes CT scans from three regions: head (neuro), chest, and abdomen.
    By specifying the 'region' parameter, it's possible to only use a subset of the data.
    Args:
        root (str, optional): The root directory of the dataset. Defaults to the current working directory.
        train (bool, optional): Whether to use the training set. Defaults to True.
        region (str, optional): The region of interest ('abdomen', 'chest', 'neuro'). Defaults to None (use all regions).
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.
        download (bool, optional): Whether to download the dataset if it's not already present in the root directory. Defaults to True.
        patch_size (int, optional): The size of patches to extract from the images. Defaults to None (do not extract patches).
        testid (list, optional): A list of subject IDs to use for testing. Defaults to ['L004'].

    Attributes:
        metadata (DataFrame): A pandas DataFrame containing metadata for the dataset.
        root (Path): The root directory of the dataset.
        data_dir (Path): The directory containing the image data.
        image_paths (list): A list of paths to the low dose images.
        target_paths (list): A list of paths to the full dose images.
        patch_size (int): The size of patches to extract from the images.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable): A function/transform that takes in the target and returns a transformed version.
    '''
    def __init__(self,
                 root=os.getcwd(),
                 train: bool = True,
                 region=None,
                 transform=None,
                 target_transform=None,
                 download=True,
                 patch_size: int | None = None):

        region_dict = {'abdomen': 'L', 'chest': 'C', 'neuro': 'N'}
        assert (region in region_dict) or (region is None)

        root = Path(root)
        if download & (not root.exists()):
            utils.download_and_extract_archive(url='<url not provided>',
                                               download_root=root)
        data_dir = root / 'LDCT-and-Projection-data'
        metadata = pd.read_csv(root / 'metadata.csv')

        if region:
            metadata = metadata[metadata['Subject ID'].apply(lambda o: o.startswith(region_dict[region]))]

        testid = metadata[metadata['Series Description'].isin(['Low Dose Images'])]['Subject ID'].iloc[:2]
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
            ld_study = metadata[(metadata['Subject ID'] == subject_id) &
                                (metadata['Series Description'] == 'Low Dose Images')]
            if len(ld_study) < 1:
                continue
            rd_study = metadata[(metadata['Subject ID'] == subject_id) &
                                (metadata['Series Description'] == 'Full Dose Images')]

            ld_path = self.root / ld_study['File Location'].item()[2:]
            rd_path = self.root / rd_study['File Location'].item()[2:]
            self.image_paths.extend(sorted(list(ld_path.glob('*.dcm'))))
            self.target_paths.extend(sorted(list(rd_path.glob('*.dcm'))))
        assert len(self.image_paths) == len(self.target_paths)

        self.patch_size = patch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        '''
        return len(self.image_paths)

    def __getitem__(self, idx):
        ld_path = self.image_paths[idx]
        rd_path = self.target_paths[idx]

        image = read_image(ld_path)
        label = read_image(rd_path)
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
    """
    A PyTorch Lightning DataModule for the Mayo Clinic Low Dose CT dataset.

    This module handles the data preprocessing, splitting into train/validation/test datasets,
    and provides dataloaders for each split.

    Args:
        data_dir (str, optional): The root directory of the dataset. Defaults to the current working directory.
        region (str, optional): The region of interest ('abdomen', 'chest', 'neuro'). Defaults to None (use all regions).
        patch_size (int, optional): The size of patches to extract from the images. Defaults to 64.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 32.
        num_workers (int, optional): The number of workers for the dataloaders. Defaults to 31.

    Attributes:
        data_dir (str): The root directory of the dataset.
        region (str): The region of interest.
        patch_size (int): The size of patches to extract from the images.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for the dataloaders.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        train_set (MayoLDGCDataset): The training dataset.
        val_set (MayoLDGCDataset): The validation dataset.
        test_set (MayoLDGCDataset): The test dataset.
        predict_set (MayoLDGCDataset): The prediction dataset.
    """

    def __init__(self, data_dir: str = "./", region=None, patch_size=64, batch_size=32, num_workers=31):
        super().__init__()
        self.data_dir = data_dir
        self.region = region
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
            train_set = MayoLDGCDataset(self.data_dir, train=True, region=self.region,
                                        patch_size=self.patch_size, transform=self.transform,
                                        target_transform=self.transform)
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
                                            region=self.region, transform=self.transform,
                                            target_transform=self.transform)

        if stage == "predict":
            self.predict_set = MayoLDGCDataset(self.data_dir, train=False,
                                               region=self.region, transform=self.transform,
                                               target_transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)


def age_to_eff_diameter(age):
    # https://www.aapm.org/pubs/reports/rpt_204.pdf
    x = torch.tensor(age)
    a = 18.788598
    b = 0.19486455
    c = -1.060056
    d = -7.6244784
    y = a + b*x**1.5 + c *x**0.5 + d*torch.exp(-x)
    eff_diam = y
    return eff_diam


def pediatric_subgroup(diameter):
    if diameter < age_to_eff_diameter(1):
        return 'newborn'
    elif (diameter >= age_to_eff_diameter(1)) & (diameter < age_to_eff_diameter(5)):
        return 'infant'
    elif (diameter >= age_to_eff_diameter(5)) & (diameter < age_to_eff_diameter(12)):
        return 'child'
    elif (diameter >= age_to_eff_diameter(12)) & (diameter < age_to_eff_diameter(22)):
        return 'adolescent'
    else:
        return 'adult'


class PediatricIQDataset(VisionDataset):
    '''
    subgroups are: [newborn, infant, child, adolescent, and adults]
    '''
    def __init__(self,
                 root=os.getcwd(),
                 train: bool=True,
                 transform=None,
                 target_transform=None,
                 download=True,
                 patch_size=None,
                 phantom=None,
                 subgroup=None,
                 testid='11.2 cm CTP404'):

      base_dir = Path(root)
      if download & (not base_dir.exists()):
        utils.download_and_extract_archive(url='https://zenodo.org/records/11267694/files/pediatricIQphantoms.zip',
                                           download_root=root)
      # build metadata file
      metadata = pd.read_csv(base_dir / 'metadata.csv').rename(columns={'Name': 'name'})
      metadata['file'] = metadata['file'].apply(lambda o: base_dir / o)

      if phantom:
        metadata = metadata[metadata.phantom == phantom]

      metadata['pediatric subgroup'] = metadata['effective diameter [cm]'].apply(pediatric_subgroup)

      if subgroup:
        if isinstance(subgroup, str):
            subgroup = [subgroup]
            metadata = metadata[metadata['pediatric subgroup'].isin(subgroup)]
      # assign slice labels
      for name in metadata.name.unique():
            for dose in metadata[metadata.name == name]['Dose [%]'].unique():
                count = len(metadata[(metadata.name == name) &
                            (metadata['Dose [%]'] == dose)])
                metadata.loc[(metadata.name == name) &
                             (metadata['Dose [%]'] == dose), 'slice'] = list(range(count))
      if train:
        metadata = metadata[metadata['name'] != testid]
      else:
        metadata = metadata[metadata['name'] == testid]
      fovs = metadata['FOV [cm]'].unique()
      self.root = base_dir
      self.metadata = metadata
      self.patch_size = patch_size
      self.ld_metadata = self.metadata[self.metadata['Dose [%]'] == 25]
      self.rd_metadata = self.metadata[self.metadata['Dose [%]'] == 100]
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):
      return len(self.ld_metadata)

    def __getitem__(self, idx):
      ld_patient = self.ld_metadata.iloc[idx]
      ld_img_path = self.root / ld_patient['file']
      image = read_image(ld_img_path)

      rd_patient = self.rd_metadata[(self.rd_metadata['name'] == ld_patient['name']) &
                                    (self.rd_metadata['slice'] == ld_patient['slice'])]
      rd_img_path = self.root / rd_patient['file'].item()
      label = read_image(rd_img_path)
      if self.transform:
        image = self.transform(image)
      if self.target_transform:
        label = self.target_transform(label)

      if self.patch_size:
        image, label = get_patch(image.squeeze(),
                                 label.squeeze(),
                                 self.patch_size)
      return image[None], label[None]

class AnthropomorphicDataset(VisionDataset):
    def __init__(self,
                 root=os.getcwd(),
                 train: bool=True,
                 transform=None,
                 target_transform=None,
                 download=True,
                 patch_size=None,
                 phantom=None,
                 subgroup=None,
                 testid='female pt151'):

      base_dir = Path(root)
      if download & (not base_dir.exists()):
        utils.download_and_extract_archive(url='https://zenodo.org/records/12538350/files/anthropomorphic.zip',
                                           download_root=root)
      # build metadata file
      metadata = pd.read_csv(base_dir / 'metadata.csv').rename(columns={'Name': 'name'})
      metadata['file'] = metadata['file'].apply(lambda o: base_dir / o)

      if phantom:
        metadata = metadata[metadata.phantom == phantom]   

      metadata['pediatric subgroup'] = metadata['effective diameter [cm]'].apply(pediatric_subgroup)

      if subgroup:
        if isinstance(subgroup, str):
            subgroup = [subgroup]
            metadata = metadata[metadata['pediatric subgroup'].isin(subgroup)]
      # assign slice labels
      for name in metadata.name.unique():
            for dose in metadata[metadata.name == name]['Dose [%]'].unique():
                count = len(metadata[(metadata.name == name) &
                            (metadata['Dose [%]'] == dose)])
                metadata.loc[(metadata.name == name) &
                             (metadata['Dose [%]'] == dose), 'slice'] = list(range(count))
      if train:
        metadata = metadata[metadata['name'] != testid]
      else:
        metadata = metadata[metadata['name'] == testid]
      fovs = metadata['FOV [cm]'].unique()
      self.root = base_dir
      self.metadata = metadata
      self.patch_size = patch_size
      self.ld_metadata = self.metadata[self.metadata['Dose [%]'] == 25]
      self.rd_metadata = self.metadata[self.metadata['Dose [%]'] == 100]
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):
      return len(self.ld_metadata)

    def __getitem__(self, idx):
      ld_patient = self.ld_metadata.iloc[idx]
      ld_img_path = self.root / ld_patient['file']
      image = read_image(ld_img_path)

      rd_patient = self.rd_metadata[(self.rd_metadata['name'] == ld_patient['name']) &
                                    (self.rd_metadata['slice'] == ld_patient['slice'])]
      rd_img_path = self.root / rd_patient['file'].item()
      label = read_image(rd_img_path)
      if self.transform:
        image = self.transform(image)
      if self.target_transform:
        label = self.target_transform(label)

      if self.patch_size:
        image, label = get_patch(image.squeeze(),
                                 label.squeeze(),
                                 self.patch_size)
      return image[None], label[None]