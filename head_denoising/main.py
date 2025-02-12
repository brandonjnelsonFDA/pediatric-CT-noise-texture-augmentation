from lightning.pytorch.cli import LightningCLI

from data import LDHeadCTDataModule
from networks import LitAutoEncoder, UNet

def cli_main():
    cli = LightningCLI(datamodule_class=LDHeadCTDataModule)

if __name__ == "__main__":
    cli_main()