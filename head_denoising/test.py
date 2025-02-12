# %%
import sys

import matplotlib.pyplot as plt
from LDHeadCTDataset import LDHeadCTDataModule
from networks import LitAutoEncoder, RED_CNN

import lightning as L

saved_path = '/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/'
dm = LDHeadCTDataModule(saved_path)
model = LitAutoEncoder()
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule=dm)
