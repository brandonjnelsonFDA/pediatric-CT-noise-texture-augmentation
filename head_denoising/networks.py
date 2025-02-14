import torch.nn.functional as F
import lightning as L
import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision

from data import LDHeadCTDataset

import sys
sys.path.append('..')
from denoising.networks import RED_CNN

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

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.torch_module = RED_CNN()

        # load sample image for tensorboard
        data_dir = '/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/'
        tfms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        test_image_set = LDHeadCTDataset(data_dir, train=False, transform=tfms, target_transform=tfms)
        self.sample = test_image_set[0]

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

        sample_image, sample_target = self.sample
        sample_image_cuda = sample_image.to('cuda')
        sample_pred = self.torch_module(sample_image_cuda).to('cpu')
        grid = torch.cat((sample_image, sample_pred, sample_target), dim=2)
        self.logger.experiment.add_image('example_images', 
                                         float_to_uint8(grid, 80, 40),
                                         batch_idx)

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
        print("Running REDCNN")
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


class UNet(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], learning_rate=1e-3):
        super(UNet, self).__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters() # Save hyperparameters for easy loading

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
        data_dir = '/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/'
        tfms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        test_image_set = LDHeadCTDataset(data_dir, train=False, transform=tfms, target_transform=tfms)
        self.sample = test_image_set[0]

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
        self.log('train_loss', loss) # Logging for TensorBoard/other loggers
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

        sample_image, sample_target = self.sample
        sample_image_cuda = sample_image.to('cuda')
        sample_pred = self(sample_image_cuda[None]).to('cpu')[0]
        grid = torch.cat((sample_image, sample_pred, sample_target), dim=2)
        self.logger.experiment.add_image('example_images', 
                                         float_to_uint8(grid, 80, 40),
                                         batch_idx)
        return loss

    def configure_optimizers(self):
        print("Running UNet")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) # Example
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) # Another example
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"} # "monitor" is important!

    def predict_step(self, batch, batch_idx):
        x = batch # No target during prediction
        y_hat = self(x)
        return y_hat


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