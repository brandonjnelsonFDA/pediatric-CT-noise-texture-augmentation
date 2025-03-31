from dotenv import load_dotenv
import lightning as L
import torch
import torch.nn as nn

load_dotenv()

HU_MEAN = -549.5100844958922  # measured from MayoLDGCDataset Chest region
HU_STD = 478.970121345292  # measured from MayoLDGCDataset Chest region


class REDCNN(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, features=96,
                 learning_rate=1e-3):
        super(REDCNN, self).__init__()
        self.mean = HU_MEAN
        self.std = HU_STD
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=5,
                               stride=1, padding=0)
        self.conv2 = nn.Conv2d(features, features, kernel_size=5,
                               stride=1, padding=0)
        self.conv3 = nn.Conv2d(features, features, kernel_size=5,
                               stride=1, padding=0)
        self.conv4 = nn.Conv2d(features, features, kernel_size=5,
                               stride=1, padding=0)
        self.conv5 = nn.Conv2d(features, features, kernel_size=5,
                               stride=1, padding=0)
        self.tconv1 = nn.ConvTranspose2d(features, features, kernel_size=5,
                                         stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(features, features, kernel_size=5,
                                         stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(features, features, kernel_size=5,
                                         stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(features, features, kernel_size=5,
                                         stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(features, out_channels, kernel_size=5,
                                         stride=1, padding=0)
        self.relu = nn.ReLU()
        self.loss_fn = nn.MSELoss()  # Example: Mean Squared Error.

    def forward(self, x):
        input_standardized = x
        x = self.standardize(x)
        # encoder
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
        # task
        out = self.destandardize(out)
        out = input_standardized - out  
        # Denoising task: subtract the noise from the input. Seems only important for REDCNN
        return out

    def standardize(self, x):
        'center data to mean 0, std 1 based on training data'
        return ((x - self.mean) / self.std)

    def destandardize(self, x):
        'undo standardization based on mean, std of training data'
        return (x * self.std + self.mean)

    def training_step(self, batch, batch_idx):
        x, y = batch  # Assuming your batch contains (input, target)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log_dict({'train_loss': loss, 'learning_rate': lr})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        print("Running UNet")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)


class UNet(L.LightningModule):
    def __init__(self, in_channels=1, out_channels=1,
                 features=[32, 64, 128, 256, 512], learning_rate=1e-3):
        super(UNet, self).__init__()
        self.mean = HU_MEAN
        self.std = HU_STD
        self.learning_rate = learning_rate
        self.save_hyperparameters()

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
            self.up_convs.append(nn.ConvTranspose2d(features[i], features[i-1],
                                 kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(features[i-1]*2, features[i-1]))

        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.loss_fn = nn.MSELoss()  # Example: Mean Squared Error.

    def forward(self, x):
        x = self.standardize(x)
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
            skip = skips[len(skips) - 2 - i // 2]  # Corresponding skip
            x = torch.cat([x, skip], dim=1)  # Concatenate
            x = self.up_convs[i+1](x)  # Double convolution

        # Output convolution
        x = self.out_conv(x)
        # task layer
        x = self.destandardize(x)
        return x

    def standardize(self, x):
        'center data to mean 0, std 1 based on training data'
        return ((x - self.mean) / self.std)

    def destandardize(self, x):
        'undo standardization based on mean, std of training data'
        return (x * self.std + self.mean)

    def training_step(self, batch, batch_idx):
        x, y = batch  # Assuming your batch contains (input, target)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log_dict({'train_loss': loss, 'learning_rate': lr})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        print("Running UNet")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
