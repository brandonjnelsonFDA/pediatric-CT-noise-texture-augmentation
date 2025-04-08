# run this first
source /home/$USER/anaconda3/envs/pedsilicoich/bin/activate
# export PHANTOM_DIRECTORY=/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/phantoms

recruit_abd abdomen_rd.toml
bash run_batchmode.sh /projects01/didsr-aiml/brandon.nelson/pediatric_CT_noise_augmentation/synthetic_data/abdomen/rd/rd.csv low_dose

recruit_abd abdomen_ld.toml
bash run_batchmode.sh /projects01/didsr-aiml/brandon.nelson/pediatric_CT_noise_augmentation/synthetic_data/abdomen/ld/ld.csv routine_dose