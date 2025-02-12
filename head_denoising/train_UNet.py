import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from data import LDHeadCTDataModule
from networks import UNet

import lightning as L


def main(saved_path):
    dm = LDHeadCTDataModule(saved_path, num_workers=5)
    model = UNet()
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = ArgumentParser("Train UNET")
    parser.add_argument('saved_path', help='directory containing training data for LDHeadCTDataModule')
    args = parser.parse_args()
    main(args.saved_path)