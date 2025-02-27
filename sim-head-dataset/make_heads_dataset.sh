# run this first
source /home/$USER/anaconda3/envs/pedsilicoich/bin/activate
export PHANTOM_DIRECTORY=/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/phantoms

recruit heads_rd.toml
bash run_batchmode.sh /projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/rd/rd.csv

recruit heads_ld.toml
bash run_batchmode.sh /projects01/didsr-aiml/brandon.nelson/pedsilicoICH/head_experiment/ld/ld.csv