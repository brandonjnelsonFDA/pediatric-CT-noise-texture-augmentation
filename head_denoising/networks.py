import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import torch.nn.functional as F
import lightning as L
import torch
import torch.nn as nn
import pydicom
from torchvision.transforms import v2
import torchvision

from data import HeadSimCTDataset, MayoLDGCDataset

load_dotenv()


def convert_to_dicom(img_slice: np.ndarray, phantom_path: str,
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


def float_to_uint8(image, window_width, window_level):
    """
    Converts a floating-point CT image (PyTorch tensor) to an 8-bit tensor using windowing.

    Args:
        image (torch.Tensor): The floating-point CT image.
        window_width (float): The window width.
        window_level (float): The window level.

    Returns:
        torch.Tensor: The 8-bit image. Returns None if inputs are invalid.
    """

    if not isinstance(image, torch.Tensor) or (image.dtype != torch.float32 and image.dtype != torch.float64):
        print("Error: Input image must be a PyTorch tensor of float32 or float64.")
        return None

    if not isinstance(window_width, (int, float)) or window_width <= 0:
        print("Error: Window width must be a positive number.")
        return None

    if not isinstance(window_level, (int, float)):
        print("Error: Window level must be a number.")
        return None

    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2

    # Clip the pixel values to the window range
    clipped_image = torch.clamp(image, lower_bound, upper_bound)

    # Scale the clipped values to the 0-255 range
    scaled_image = (clipped_image - lower_bound) / (upper_bound - lower_bound) * 255

    # Convert to uint8
    uint8_image = scaled_image.to(torch.uint8)

    return uint8_image


class REDCNN(L.LightningModule):
    def __init__(self):
        super().__init__(self, in_channels=1, out_channels=1, features=96, norm_range_min=-1024, norm_range_max=3072, learning_rate=1e-3, output_dir=None)
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(features, features, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(features, features, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(features, features, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(features, features, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(features, features, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(features, features, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(features, features, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(features, features, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(features, out_channels, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

        # load sample image for tensorboard
        tfms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        head_CT_sim_test = HeadSimCTDataset(os.environ['HEAD_CT_PATH'], train=False,
                                            transform=tfms, target_transform=tfms)
        mayo_test = MayoLDGCDataset(os.environ['LDGC_PATH'], train=False,
                                    transform=tfms, target_transform=tfms)
        self.sample_head = head_CT_sim_test[54]
        self.sample_mayo = mayo_test[248]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.torch_module(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)

        # Log the learning rate
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log_dict({'train_loss': loss, 'learning_rate': lr}) # Logging for TensorBoard/other loggers

        return loss

    def test_and_log_image(self, test_pair, title, batch_idx, wwwl=(80,40)):
        # Ensure the test image is a torch tensor and move it to the device
        test_image, test_label = test_pair
        prediction = self(test_image[None].to(self.device)).to('cpu')[0]
        # Concatenate the test image and the prediction for visualization
        grid = torch.cat((test_image, prediction, test_label), dim=2)
        # Convert the grid to uint8 and log it on TensorBoard
        ww, wl = wwwl
        self.logger.experiment.add_image(title, float_to_uint8(grid, ww, wl), batch_idx)

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self(x)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.test_and_log_image(self.sample_mayo, 'mayo LDGC', batch_idx, (350, 50))
        self.test_and_log_image(self.sample_head, 'sim CT head', batch_idx, (80, 40))

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x_hat = self.torch_module(x)
        test_loss = F.mse_loss(x_hat, y)
        self.log("test_loss", test_loss)

    def forward(self, x):
        # encoder
        x = self.normalize(x)
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        out = self.denormalize(out)
        return out

    def normalize(self, image):
        image = (image - self.norm_range_min) / (self.norm_range_max - self.norm_range_min)
        return image

    def denormalize(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image
  
    def predict_step(self, batch, batch_idx):
        x, y = batch # No target during prediction
        predictions = self(x)
        for i, prediction in enumerate(predictions):
            # convert tensor to numpy array
            prediction = prediction.detach().cpu().numpy()
            convert_to_dicom(prediction, self.output_dir / 'prediction' / f'redcnn_{batch_idx:03d}_{i:03d}.dcm')
            convert_to_dicom(x[i].detach().cpu().numpy(), self.output_dir / 'input' / f'input_{batch_idx:03d}_{i:03d}.dcm')
            convert_to_dicom(y[i].detach().cpu().numpy(), self.output_dir / 'target' / f'target_{batch_idx:03d}_{i:03d}.dcm')
        return prediction

    def configure_optimizers(self):
        print("Running REDCNN")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class UNet(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], learning_rate=1e-3, output_dir=None):
        super(UNet, self).__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters() # Save hyperparameters for easy loading
        self.output_dir = output_dir

        self.in_conv = DoubleConv(in_channels, features[0])
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path
        for i in range(len(features) - 1):
            self.down_convs.append(DoubleConv(features[i], features[i+1]))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1])

        # Upward path
        for i in range(len(features) - 1, 0, -1):
            self.up_convs.append(nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(features[i-1]*2, features[i-1]))

        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.loss_fn = nn.MSELoss()  # Example: Mean Squared Error.  Consider other losses like SSIM or a combination.

        # load sample image for tensorboard
        tfms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        head_CT_sim_test = HeadSimCTDataset(os.environ['HEAD_CT_PATH'], train=False,
                                            transform=tfms, target_transform=tfms)
        mayo_test = MayoLDGCDataset(os.environ['LDGC_PATH'], train=False,
                                    transform=tfms, target_transform=tfms)
        self.sample_head = head_CT_sim_test[54]
        self.sample_mayo = mayo_test[248]

    def forward(self, x):
        # Downward path
        skips = []
        x = self.in_conv(x)
        skips.append(x)  # Skip connection for the first level

        for down_conv in self.down_convs:
            x = self.pool(x)
            x = down_conv(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upward path
        for i in range(0, len(self.up_convs), 2):
            x = self.up_convs[i](x)  # Transpose convolution
            skip = skips[len(skips) - 2 - i // 2]  # Corresponding skip connection
            x = torch.cat([x, skip], dim=1)  # Concatenate
            x = self.up_convs[i+1](x)  # Double convolution

        # Output convolution
        x = self.out_conv(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch  # Assuming your batch contains (input, target)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log_dict({'train_loss': loss, 'learning_rate': lr}) # Logging for TensorBoard/other loggers
        return loss

    def test_and_log_image(self, test_pair, title, batch_idx, wwwl=(80,40)):
        # Ensure the test image is a torch tensor and move it to the device
        test_image, test_label = test_pair
        prediction = self(test_image[None].to(self.device)).to('cpu')[0]
        # Concatenate the test image and the prediction for visualization
        grid = torch.cat((test_image, prediction, test_label), dim=2)
        # Convert the grid to uint8 and log it on TensorBoard
        ww, wl = wwwl
        self.logger.experiment.add_image(title, float_to_uint8(grid, ww, wl), batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

        self.test_and_log_image(self.sample_mayo, 'mayo LDGC', batch_idx, (350, 50))
        self.test_and_log_image(self.sample_head, 'sim CT head', batch_idx, (80, 40))
        return loss

    def configure_optimizers(self):
        print("Running UNet")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) # Example
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) # Another example
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"} # "monitor" is important!

    def predict_step(self, batch, batch_idx):
        x, y = batch # No target during prediction
        predictions = self(x)
        for i, prediction in enumerate(predictions):
            # convert tensor to numpy array
            prediction = prediction.detach().cpu().numpy()
            convert_to_dicom(prediction, self.output_dir / 'prediction' / f'unet_{batch_idx:03d}_{i:03d}.dcm')
            convert_to_dicom(x[i].detach().cpu().numpy(), self.output_dir / 'input' / f'input_{batch_idx:03d}_{i:03d}.dcm')
            convert_to_dicom(y[i].detach().cpu().numpy(), self.output_dir / 'target' / f'target_{batch_idx:03d}_{i:03d}.dcm')
        return prediction


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)