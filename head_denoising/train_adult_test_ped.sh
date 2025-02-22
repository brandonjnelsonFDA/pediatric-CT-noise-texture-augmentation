SAVE_DIR=lightning_logs/redcnn_mayo_chest
MODEL=REDCNN
# train on adult data
python main.py fit     --model $MODEL\
                       --trainer.callbacks DicomWriter\
                       --trainer.callbacks.write_interval epoch\
                       --trainer.callbacks.output_dir $SAVE_DIR\
                       --trainer.max_epochs 1\
                       --trainer.logger TensorBoardLogger\
                       --trainer.logger.save_dir $SAVE_DIR\
                       --data MayoLDGCDataModule\
                       --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                       --data.region chest\
                       --data.num_workers 11\
# test on pediatric data
python main.py predict --model $MODEL\
                       --trainer.callbacks DicomWriter\
                       --trainer.callbacks.write_interval epoch\
                       --trainer.callbacks.output_dir $SAVE_DIR/PedIQ\
                       --trainer.logger TensorBoardLogger\
                       --trainer.logger.save_dir $SAVE_DIR/PedIQ\
                       --data PediatricIQDataModule\
                       --data.data_dir /projects01/didsr-aiml/brandon.nelson/pediatric_CT_noise_augmentation\
                       --data.num_workers 11\
                       --ckpt_path last

# test on adult data
python main.py predict --model $MODEL\
                       --trainer.callbacks DicomWriter\
                       --trainer.callbacks.write_interval epoch\
                       --trainer.callbacks.output_dir $SAVE_DIR/MayoLDGC\
                       --trainer.logger TensorBoardLogger\
                       --trainer.logger.save_dir $SAVE_DIR/MayoLDGC\
                       --data MayoLDGCDataModule\
                       --data.data_dir /projects01/didsr-aiml/common_data/Mayo_CT_LDGC\
                       --data.num_workers 11\
                       --ckpt_path last
