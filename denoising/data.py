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
            train_set_size = int(len(train_set) * 0.9)
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
    A PyTorch dataset for the Pediatric Image Quality CT dataset.

    This dataset includes CT scans from four phantoms: CTP404, MITA-LCD, uniform water, and anthropomorphic.
    By specifying the 'region' parameter, it's possible to only use a subset of the data.
    Args:
        root (str, optional): The root directory of the dataset. Defaults to the current working directory.
        train (bool, str, optional): Whether to use the training set. Defaults to 'predict'.
        phantom (str, optional): Phantom type to include ('CTP404', 'MITA-LCD', 'uniform', 'anthropomorphic'). Defaults to None (use all phantoms).
        subgroup (str, list, optional): newborn, child, adolescent, adult
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.
        download (bool, optional): Whether to download the dataset if it's not already present in the root directory. Defaults to True.
        patch_size (int, optional): The size of patches to extract from the images. Defaults to None (do not extract patches).

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
                 train: bool|str='predict',
                 phantom=None,
                 subgroup=None,
                 transform=None,
                 target_transform=None,
                 download=True,
                 patch_size=None):

      base_dir = Path(root)
      if download & (not base_dir.exists()):
        utils.download_and_extract_archive(url='https://zenodo.org/records/11267694/files/pediatricIQphantoms.zip',
                                           download_root=root / 'pediatricIQphantoms')
        utils.download_and_extract_archive(url='https://zenodo.org/records/12538350/files/anthropomorphic.zip',
                                           download_root=root / 'anthropomorphic')
      # build metadata file
      dfs = []
      for series in ['pediatricIQphantoms', 'anthropomorphic']:
        temp_dir = base_dir / series
        temp = pd.read_csv(temp_dir / 'metadata.csv').rename(columns={'Name': 'name'})
        temp['file'] = temp['file'].apply(lambda o: temp_dir / o)
        dfs.append(temp)
      metadata = pd.concat(dfs, ignore_index=True)

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

      testid = metadata['name'].iloc[:2]
      if train == 'predict':
        pass
      elif train == True:
        metadata = metadata[~metadata['name'].isin(testid)]
      elif train == False:
        metadata = metadata[metadata['name'].isin(testid)]
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
      return image, label


class PediatricIQDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Pediatric Image Quality CT dataset.

    This module handles the data preprocessing, splitting into train/validation/test datasets,
    and provides dataloaders for each split.

    Args:
        data_dir (str, optional): The root directory of the dataset. Defaults to the current working directory.
        phantom (str, optional): Phantom type to include ('CTP404', 'MITA-LCD', 'uniform', 'anthropomorphic'). Defaults to None (use all phantoms).
        subgroup (str, list, optional): newborn, child, adolescent, adult based on size
        patch_size (int, optional): The size of patches to extract from the images. Defaults to 64.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 32.
        num_workers (int, optional): The number of workers for the dataloaders. Defaults to 31.

    Attributes:
        data_dir (str): The root directory of the dataset.
         (str): The region of interest.
        patch_size (int): The size of patches to extract from the images.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for the dataloaders.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        train_set (MayoLDGCDataset): The training dataset.
        val_set (MayoLDGCDataset): The validation dataset.
        test_set (MayoLDGCDataset): The test dataset.
        predict_set (MayoLDGCDataset): The prediction dataset.
    """

    def __init__(self, data_dir: str = "./", phantom=None, subgroup=None, shuffle=False, patch_size=64, batch_size=32, num_workers=31):
        super().__init__()
        self.data_dir = data_dir
        self.phantom = phantom
        self.subgroup = subgroup
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

    def prepare_data(self):
        PediatricIQDataset(self.data_dir)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_set = PediatricIQDataset(self.data_dir, train=True, phantom=self.phantom,
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
            self.test_set = PediatricIQDataset(self.data_dir, train=False,
                                               region=self.region, transform=self.transform,
                                               target_transform=self.transform)

        if stage == "predict":
            self.predict_set = PediatricIQDataset(self.data_dir, train='predict',
                                                  phantom=self.phantom, transform=self.transform,
                                                  target_transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)


class AugmentedDataSet(VisionDataset):
    '''
    A PyTorch dataset that uses one dataset (dset1) as the main dataset and uses a second dataset (dset2) for adding noise
    to augment the data.

    The dataset is used in the context of low dose CT image enhancement by adding noise from a separate dataset
    to the input images. This can help improve the model's ability to reconstruct high quality images from low dose data.

    Args:
        dset1 (VisionDataset): The main dataset to use for the input images.
        dset1_kwargs (dict): The arguments to pass to the constructor of dset1.
        dset2 (VisionDataset): The dataset to use for adding noise to the input images. Should be a low dose CT dataset with corresponding high dose images.
        dset2_kwargs (dict): The arguments to pass to the constructor of dset2.
        proportion (float, optional): The probability of adding noise to an input image. Defaults to 0.5.

    Attributes:
        dset1 (VisionDataset): The main dataset to use for the input images.
        dset2 (VisionDataset): The dataset to use for adding noise to the input images.
        root (Path): The root directory of the input dataset.
        proportion (float): The probability of adding noise to an input image.
    '''
    def __init__(self, dset1: VisionDataset, dset1_kwargs: dict, dset2: VisionDataset, dset2_kwargs: dict, proportion: float=0.5):
        self.dset1 = dset1(**dset1_kwargs)
        self.dset2 = dset2(**dset2_kwargs)
        self.root = dset1_kwargs['root']
        self.proportion = proportion

    def __len__(self):
        '''
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        '''
        return len(self.dset1)

    def __getitem__(self, i):
        '''
        Returns the sample and target at the given index. If a random value is less than the predefined proportion, then the input image
        is augmented with noise from the second dataset.

        Args:
            i (int): The index of the sample.

        Returns:
            tuple: A tuple containing the sample and target.
        '''
        image, label = self.dset1[i]
        if torch.rand(1)[0] < self.proportion:
            idx = torch.randint(0, len(self.dset2), size=(1,))[0].numpy()
            x2, y2 = self.dset2[idx]
            noise = y2 - x2
            image = label + noise
        return image, label


class AugmentedDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule that uses a primary dataset and a secondary dataset to generate augmented data.
    The secondary dataset is used to add noise to the inputs from the primary dataset, thus providing a method to
    artificially increase the amount of available data.

    This DataModule is designed for use with low-dose CT image enhancement tasks, where adding noise to the
    input images can help improve the model's ability to reconstruct high-quality images from low-dose data.

    Args:
        dataset1 (VisionDataset or str): The primary dataset or its string name to be used for data retrieval.
        dataset1_kwargs (dict): A dictionary of keyword arguments for the primary dataset constructor.
        dataset2 (VisionDataset or str): The secondary dataset or its string name to be used for adding noise.
        dataset2_kwargs (dict): A dictionary of keyword arguments for the secondary dataset constructor.
        proportion (float, optional): The probability of using an augmented data sample. Defaults to 0.5.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        patch_size (int, optional): The size of patches to extract from the images. Defaults to 64.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 32.
        num_workers (int, optional): The number of workers for the dataloaders. Defaults to 31.

    Attributes:
        dataset1 (VisionDataset): The primary dataset.
        dataset1_kwargs (dict): A dictionary of keyword arguments for the primary dataset.
        dataset2 (VisionDataset): The secondary dataset.
        dataset2_kwargs (dict): A dictionary of keyword arguments for the secondary dataset.
        proportion (float): The proportion of using an augmented data sample.
        shuffle (bool): Whether to shuffle the data.
        patch_size (int): The size of patches to extract from the images.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for the dataloaders.
        transform (callable): A function/transform that takes in a sample and returns a transformed version.
        train_set (AugmentedDataSet): The training dataset.
        val_set (AugmentedDataSet): The validation dataset.
        test_set (AugmentedDataSet): The test dataset.
        predict_set (AugmentedDataSet): The prediction dataset.
    """

    def __init__(self, dataset1, dataset1_kwargs, dataset2, dataset2_kwargs, proportion=0.5, shuffle=True, patch_size=64, batch_size=32, num_workers=31):
        super().__init__()
        if isinstance(dataset1, str):
            dataset1 = eval(dataset1)
        if isinstance(dataset2, str):
            dataset2 = eval(dataset2)
        self.dataset1 = dataset1
        self.dataset1_kwargs = dataset1_kwargs
        self.dataset2 = dataset2
        self.dataset2_kwargs = dataset2_kwargs
        self.proportion = proportion
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])

        self.dataset1_kwargs['patch_size'] = patch_size
        self.dataset1_kwargs['transform'] = self.transform
        self.dataset1_kwargs['target_transform'] = self.transform
        self.dataset2_kwargs['patch_size'] = patch_size
        self.dataset2_kwargs['transform'] = self.transform
        self.dataset2_kwargs['target_transform'] = self.transform

    def prepare_data(self):
        self.dataset1(**self.dataset1_kwargs)
        self.dataset2(**self.dataset2_kwargs)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset1_kwargs['train'] = True
            self.dataset2_kwargs['train'] = True
            train_set = AugmentedDataSet(dset1=self.dataset1, dset1_kwargs=self.dataset1_kwargs,
                                         dset2=self.dataset2, dset2_kwargs=self.dataset2_kwargs,
                                         proportion=self.proportion)
            # use 80% of training data for actual training and 20% for validation
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
            self.dataset1_kwargs['train'] = False
            self.dataset2_kwargs['train'] = False
            self.test_set = AugmentedDataSet(dset1=self.dataset1, dset1_kwargs=self.dataset1_kwargs,
                                             dset2=self.dataset2, dset2_kwargs=self.dataset2_kwargs,
                                             proportion=self.proportion)

        if stage == "predict":
            self.dataset1_kwargs['train'] = False
            self.dataset2_kwargs['train'] = False
            self.predict_set = AugmentedDataSet(dset1=self.dataset1, dset1_kwargs=self.dataset1_kwargs,
                                             dset2=self.dataset2, dset2_kwargs=self.dataset2_kwargs,
                                             proportion=self.proportion)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)