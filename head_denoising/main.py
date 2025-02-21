from lightning.pytorch.cli import LightningCLI

from data import HeadSimCTDataModule, MayoLDGCDataModule
from networks import REDCNN, UNet
from callbacks import DicomWriter
# https://github.com/Lightning-AI/pytorch-lightning/blob/3dcf7130c554f4511c756ccbb4e3a417103d595d/pytorch_lightning/loggers/tensorboard.py#L110-L119
def cli_main():
    pred_writer = DicomWriter(output_dir="pred_path", write_interval="epoch")
    trainer = Trainer(callbacks=[pred_writer])
    cli = LightningCLI(trainer_class=trainer)

if __name__ == "__main__":
    cli_main()