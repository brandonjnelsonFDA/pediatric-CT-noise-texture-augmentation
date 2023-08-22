Start by

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

### Conda
conda create -n peds_aug_tensorflow --file requirements.txt -y -c simpleitk
conda activate peds_aug_tensorflow

### Pip


## Steps

## TODOs
1. try only augmenting with specific diameters, 1 that is only newborns (112mm), one that is midrange, and only adults and see how that compares to mixing all of them together
2. try transfer learning,
3. 
FAQS

Octave
If installing and running a fresh Octab


Installing Image on Octave
pkg: please install the Debian package "liboctave-dev" to get the mkoctfile command:
sudo apt-get install liboctave-dev

Installing TexLive on Linux
https://tug.org/texlive/quickinstall.html#running