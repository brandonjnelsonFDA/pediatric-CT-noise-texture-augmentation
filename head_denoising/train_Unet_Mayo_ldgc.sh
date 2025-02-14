python main.py fit --model UNet\
                   --data MayoLDGCDataModule\
                   --data.num_workers 5\
                   --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                   --trainer.max_epochs 100