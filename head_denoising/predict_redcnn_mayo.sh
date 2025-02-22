SAVE_DIR=lightning_logs/redcnn_mayo_chest
python main.py fit     --trainer.callbacks.output_dir $SAVE_DIR\
                       --model REDCNN\
                       --data MayoLDGCDataModule\
                       --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                       --trainer.callbacks.write_interval epoch\
                       --data.region chest\
                       --data.num_workers 11\
                       --trainer.logger TensorBoardLogger\
                       --trainer.logger.save_dir $SAVE_DIR

python main.py predict --trainer.callbacks DicomWriter\
                       --trainer.callbacks.output_dir $SAVE_DIR\
                       --model REDCNN\
                       --data MayoLDGCDataModule\
                       --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                       --trainer.callbacks.write_interval epoch\
                       --data.region chest\
                       --data.num_workers 11\
                       --trainer.logger TensorBoardLogger\
                       --trainer.logger.save_dir $SAVE_DIR