
1. [X] downloading the pretrained denoising models from the [Denoising tutorial](https://colab.research.google.com/drive/1N8V56eHEx3uIWIahBvRGAorszAziyAs7#scrollTo=FxrP4SiMdmUT)
    - might need to add a model save option
2. [X] download the [LCD toolkit and dataset](https://github.com/DIDSR/LCD_CT)
3. [X] Use the downloaded denoisers to process the CCT189 images
4. [X] Use the LCD toolkit to evaluate the two models trained in the tutorial (see if VGG actually made a difference)
5. save that tutorial as a new "advanced tutorial" for the LCD toolkit
6. [X] next get a baseline performance of these models with PED-ETK
7. start developing data aug and compare reevaluate comparing against previous baselines

- ready on [google colab](https://colab.research.google.com/drive/1aYFFunBcIK2D98qPEmMVqO98uVWepziW#scrollTo=Zt9LBQdAHfYy), need to bring here

![Alt text](LCD_results.png)
These are the preliminary results for the standard LCD CT (step 4)

## Install

jupyterlab via port forwarding
<https://thedatafrog.com/en/articles/remote-jupyter-notebooks/>

pip3 install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu118>

### Conda

conda create -n peds_aug_tensorflow --file requirements.txt -y -c simpleitk
conda activate peds_aug_tensorflow

### Pip

## Steps

FAQS

Common errors:
Octave

If installing and running a fresh Octab

Installing Image on Octave `pkg install -forge image`

```
    /bin/bash: /home/brandon.nelson/miniconda3/envs/peds_data_aug/lib/libtinfo.so.6: no version information available (required by /bin/bash)
    /bin/bash: /home/brandon.nelson/miniconda3/envs/peds_data_aug/lib/libtinfo.so.6: no version information available (required by /bin/bash)
    configure: error: in `/tmp/oct-6zg6LA/image-2.14.0/src':
    configure: error: C++ compiler cannot create executables
    See `config.log' for more details
    checking for a sed that does not truncate output... /usr/bin/sed
    checking for octave... /home/brandon.nelson/miniconda3/envs/peds_data_aug/bin/octave-7.3.0
    checking for mkoctfile... /home/brandon.nelson/miniconda3/envs/peds_data_aug/bin/mkoctfile-7.3.0
    checking whether the C++ compiler works... no

    error: pkg: error running the configure script for image
    error: called from
        configure_make at line 101 column 9
        install at line 202 column 
```

pkg: please install the Debian package "liboctave-dev" to get the mkoctfile command:
sudo apt-get install liboctave-dev

Installing TexLive on Linux
<https://tug.org/texlive/quickinstall.html#running>

## LDGC

L067, L096, L109, L143 (non con), L192 (con), L286, L291 (con), L310 (con), L333, L506

## TODO

1. [X] focus on using open source implementation of redcnn training and then add augmentation there and evaluate with pipeline <<< current activity, when training finished work on
2. [X] double check NPS results
3. [X] measure denoising efficiency across all phantom sizes similar to [iq_phantom_validation.py](https://github.com/bnel1201/Ped-ETK/blob/main/evaluation/iq_phantom_validation.py)
4. [ ] build pediatric only model --> peds train/test to get upper bound
5. [ ] try only augmenting with specific diameters, 1 that is only newborns (112mm), one that is midrange, and only adults and see how that compares to mixing all of them together\
6. [ ] add more models [https://github.com/prabhatkc/ct-recon/tree/main/Denoising/DLdenoise](Prabhat's DLdenoise repo) --> *UNET in particular*

Desired output directory structure:
(anthropomorphic does this: /gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/anthropomorphic) but not CCT189 yet

```directory
phantom /
        / diameter /
                    / sa, sp
                            / dose /
                                   / recon
```

make sure all headers have correct pixel sizes

Notebook Layout
---------------

Ideally these notebooks will import the main code so as to prevent multiple versions from floating around and will correspond to different sections of the paper

Method development:

Consider absorbing make_noise_patches.ipynb into characterizing_noise_augmentation.ipynb to have 1 method dev notebook.

- [X] characterizing noise properties in patient data
  a. adult training data
  b. peds testing data
  c. compare noise properties in peds xcats vs adult xcats and confirm they agree with phantoms of equal sizes
- [X] augmentation development
  a. inspecting phantom scans and noise images
  b. patch generation
  c. comparing patch noise properties with adult training and pediatric testing data
- [x] physical scan validation

Evaluation Results:

- [X] Denoising efficiency
  - [x] noise magnitude reduction in uniform phantoms
  - [x] noise magnitude redcucion in anthropomorphic phantoms
  - [ ] noise magnitude reduction in adult patient images (no peds patient images)
  - [x] RMSE reduction in uniform phantoms
  - [x] RMSE in anthropomorphic phantoms
  
- [X] Sharpness preservation
 a. MTF plots
- [X] Noise texture preservation
 a. Uniform phantom images and noise difference images
 b. NPS plots before and after denoising
- [X] Task performance
 a. low contrast detectability
