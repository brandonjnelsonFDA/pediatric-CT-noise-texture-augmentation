import torch.nn.functional as F
import lightning as L
import torch
from torchvision.transforms import v2
import torchvision

from LDHeadCTDataset import LDHeadCTDataset

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
