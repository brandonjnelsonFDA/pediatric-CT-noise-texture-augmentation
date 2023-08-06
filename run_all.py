# this tutorial trains 2 CNN denoising models on a very small dataset and downloads the results from a REDCNN trained on a larger (but still small dataset) then tests them on the LCD-CT CCT189 phantom simulated on a Siemens Somatom scanner. The denoised images from each model are then evaluated in terms of low contrast detectability across dose using the LCD-CT tool.
python modified_cnndenoisingtutorial_magiciancorner

python process_CCT189

# git clone https://github.com/DIDSR/LCD_CT.git

octave-launch --no-gui measure_denoised_LCD.m

python plot_results