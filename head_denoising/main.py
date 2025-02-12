from lightning.pytorch.cli import LightningCLI

from data import LDHeadCTDataModule
from networks import LitAutoEncoder

def cli_main():
    cli = LightningCLI(LitAutoEncoder, LDHeadCTDataModule)

if __name__ == "__main__":
    cli_main()