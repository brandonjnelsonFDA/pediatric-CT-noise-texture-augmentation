from lightning.pytorch.cli import LightningCLI

from data import LDHeadCTDataModule, MayoLDGCDataModule
from networks import LitAutoEncoder, UNet

def cli_main():
    cli = LightningCLI()

if __name__ == "__main__":
    cli_main()