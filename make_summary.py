#%%
from pylatex import Document, Section, Subsection, Figure, SubFigure, NoEscape, Command
from pylatex.labelref import Label

from pathlib import Path
from argparse import ArgumentParser

# %%
def main(results_dir, notes=None):
    results_dir = Path(results_dir)
    geometry_options = {"tmargin":"1in", "bmargin":"1in", "lmargin":"1in", "rmargin":"1in"}
    doc = Document(geometry_options=geometry_options)

    doc.preamble.append(Command('title', 'Noise Texture Augmentation to Improve Generalizability of Deep Learning Denoising in Pediatric CT'))
    doc.preamble.append(Command('author', 'Brandon J. Nelson, Prabhat Kc, Andreu Badal, Lu Jiang, Rongping Zeng'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    diameters = [112, 151, 292]

    if notes:
        with doc.create(Section('Experiment Details')):
            doc.append(notes)

    with doc.create(Section('Introduction')):
        doc.append(NoEscape(r'''
Deep learning image reconstruction and denoising have been shown to be a viable option for reducing noise in CT imaging, enabling dose reductions on par with or potentially greater than the previous state of the art model-based iterative reconstruction. In addition deep learning denoising, requires less computation at inference time and can better preserve noise texture relative to filtered backprojection (FBP), a feature generally favored by radiologists and known to affect low contrast detectability (**cite**).\n
        
However, a key limitation of deep learning techniques, is their limited ability to generalize to data characteristically different from their training data. Prior studies have shown such deep learning CT denoising models are particularly sensitive to changes in noise texture of  the input image prior to denoising. These changes in noise texture can come from input images being reconstructed with different reconstruction kernels\cite{zengPerformanceDeepLearningbased2022} and reconstructed fields of view (FOV).\cite{huberEvaluatingConvolutionalNeural2021} Such changes in noise texture are common when imaging pediatric patients which can be particularly smaller than adults, particularly in abdominal imaging, where FOV is routinely adapted to fit the patient. This effect has recently been shown to have substantial influence when considering using deep learning denoising models trained on adult patients but applied to pediatric patients. Given that pediatric patients can be considerably smaller than adult patient, it was shown that the smaller associated FOV results in reduction in denoising performance that progressively worsened as patient size and FOV decreased relative to the training distribution.\cite{nelsonPediatricSpecificEvaluationsDeep2023}\n
        
This demonstration that adult-trained deep learning models do not generalize to pediatric patient raises health equity concerns as it could limit access for pediatric patients to the latest and greatest medical advancements made available with deep learning. Pediatric patients are under represented in radiological imaging, making up only 5\% of scans despite representing 20\% of the US population.\cite{smith-bindmanTrendsUseMedical2019} As a result deep learning models are generally not trained with pediatric data. The goal of this work is to leverage data augmentation, a deep learning model training technique for enhancing limited training datasets, as a means to improve deep learning CT denoising to patients of sizes outside their training distribution. This is done by extracting noise textures extracted from scans of phantoms representative of pediatric sizes and FOVs to augment the adult traininging data. The result is a denoising model that generalizes better to smaller patients, saving time, resources, and radiation exposure compared to compiling large datasets with these patients, which is generally not feasible.
'''))

    with doc.create(Section('Methods')):
        doc.append(
'''
In this work, noise texture patches are generated from simulated CT scans of phantoms representative of different pediatric waist diameters ranging from newborn to adolescent using body fitting FOVs. These patches are then used to augment the training of a deep learning denoising model using a training dataset consisting of adult patient data. The magnitude of noise reduction, noise texture preservation, image sharpness, and low contrast detectability using model observers were all used to assess whether the  proposed augmentation technique improved generalizability in smaller patients.

''')
        with doc.create(Figure(position='h!')) as fig:
            with doc.create(SubFigure(
                                position='b',
                                width=NoEscape(r'0.4\linewidth'))) as subfig:
                image_filename = results_dir/'standard_training_schematic.png'
                subfig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.9\linewidth'))
                subfig.add_caption('Standard Model Training')
            with doc.create(SubFigure(
                                position='b',
                                width=NoEscape(r'0.5\linewidth'))) as subfig:
                image_filename = results_dir/'augmented_training_schematic.png'
                subfig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                subfig.add_caption('Augmented Model Training')
            fig.add_caption('Schematic diagram of proposed data augmentation. (a) Standard model training where training input patches are given to a model to make a prediction that is compared against the training target using the loss function used to update the model. (b) In the augmented training noise patches are added to the low noise training target to make a new augmented input. While training the model some proportion of training uses the augmented input while the remainder uses the original training inputs. This augmentation frequency is an additional training hyperparameter introduced by this method.')
            fig.append(Label('fig:schematic'))

        with doc.create(Subsection('Making Noise Texture Patches')):
            doc.append(NoEscape(
r'''
Water cylinders of different sizes were numerically simulated with CT projection data simulated using the Michigan Image Reconstruction Toolbox (MIRT). The acquisition parameters were modeled after the Siemens Sensation scanner with noise texture, sharpness, mA and kVp matching those used in the Mayo Clinic's Low Dose Grand Challenge Dataset.\cite{mccolloughLowdoseCTDetection2017} CT images were reconstructed from this projection data using fitting FOVs equal to 110\% the cylinder diameter as shown in Figure \ref{fig:methods}a. CTDIvol in the simulations was scaled by adjusting mA in the simulations such that the noise, measured as the standard deviation of voxel intensities in a region of interest, were approximately constant for all phantom sizes.\cite{nelsonPediatricSpecificEvaluationsDeep2023}
'''))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'methods.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                fig.add_caption('Creating noise patches of varying texture for data augmentation. (a) Water phantoms of varying diameters (112, 185, and 216 mm are shown) are virtually scanned, [mean, standard deviation] are shown for different 30x30 pixel patches from different regions of the image. Taking the difference of multiple repeat scans with different instances of projection Poisson noise yields the noise only images (b). (c) patches taken from different regions around the noise images show different noise orientations. Noise grain size also decreases with increasing phantom size. (d) 2D Noise power spectra illustrate different orientations and spatial frequencies of noise between patches taken from diffent regions and phantom sizes.')
                fig.append(Label('fig:methods'))

            doc.append(NoEscape(r'''
Noise only images from each size cylinder phantom image were then made by taking the dfference of repeat scans with different projection noise instances (Figure \ref{fig:methods}b). Note that the measured noise from central and peripheral regions is approximately $\sqrt{2\sigma^2}$, where $\sigma$ is the measured noise from the reconstructed images in Figure \ref{fig:methods}b because we defined noise as the standard deviation, rather than the variance. Patches were then selected from random locations across these noise only images (Figure \ref{fig:methods}c). The noise only images are split into patches since most denoising models are trained on image patches rather than whole image slices (**cite**). The matrix size of these random patches is set to match the matrix size of the training set image patches. By selecting random locations from the noise images, these noise patches contain different orientations of noise, shown by the different orientation of NPS images taken from the center, top and left of the noise images (Figure \ref{fig:methods}d). The noise patches from different sized phantoms also contain noise of varying grain size. Noise patches from the smaller diameter phantoms have larger noise grains and thus their noise power spectra are predominantly lower frequency, while the smaller noise grain patches from large FOV phantom scans are higher frequency (Figure \ref{fig:methods}d).'''))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'trainingnoise.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.6\linewidth'))
                fig.add_caption('Comparison of noise textures found in training dataset compared to those to be added via data augmentation. (a) Example training inputs. (b) Noise textures estimated by taking the difference between training inputs and targets. (c) The averaged noise power spectra (NPS) across noise textures found in training. (d) Averged NPS across generated noise textures from each diameter phantom to be used for augmented training.')
                fig.append(Label('fig:trainingnoise'))

            doc.append(NoEscape(r'''
Compared to noise images from the training set (Figure \ref{fig:trainingnoise}b), found by taking the difference between training inputs (Figure \ref{fig:trainingnoise}a) and training targets, the phantom simulated noise patches encompasss a wider range of noise spatial frequencies than encountered in the adult-only training set as shown by comparing the averaged training noise NPS in Figure \ref{fig:trainingnoise}c with that of the generated noise patches to be used for augmentation shown in Figure \ref{fig:trainingnoise}d.\n
'''))

        with doc.create(Subsection('Denoising Model Training with Augmentation')):
            doc.append(
'''
The goal of our proposed size-based noise data augmentation is to incorporate this diversity of noise textures into the model training loop to improve the model generalizability to remove noise from a wider range of noise textures as would be seen in smaller patients and pediatric patients. This is illustrated in diagram X, where every X percent of training examples, the usual low dose input, high dose training pair is replaced with a new augmented training pair. The new input is the high dose input with an added random noise patch where the ta 
''')

        with doc.create(Subsection('Size Generalization Evaluations')):
            doc.append(NoEscape(
r'''
To assess whether the data augmentation was able to enhance the generalizability model to different sized phantoms, evautions of noise reduction, sharpness, noise texture, and low contrast detectability were performed.\n
'''))
    
    with doc.create(Section('Results')):

        with doc.create(Subsection('Noise Reduction')):
            doc.append("Our first assessment compares noise standard deviation measured across phantom diameter. Noise standard deviation is a simple measure of overall noise magnitude which can be usefull for assessing noise reduction. However noise standard deviation does not account for noise texture which can affect the ability of a reader to detect low contrast lesions.")

            doc.append(NoEscape(r"Figure \ref{fig:stdnoise} tracks the absolute noise level (measured as standard deviation) as a function of phantom diameter defined as"))
            doc.append(NoEscape(r'$\sigma - \sigma_{FBP}$'))
            doc.append(NoEscape(r'this means that a lower noise in the processed image will yield a more negative $\Delta std$'))
            doc.append(NoEscape(r"Figure \ref{fig:noisereduction} then tracks the \emph{noise reduction} relative to FBP. Noise reduction is here defined as"))
            doc.append(NoEscape(r'100% \times \sigma_{FBP} - \sigma)/\sigma_{FBP}'))
            doc.append('Note that with this definition as the noise standard deviation approaches 0, the noise reduction approaches 100%.')

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'std_noise.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('Measured std noise across phantom sizes')
                fig.append(Label('fig:stdnoise'))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'noise_reduction.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                fig.add_caption('Reduction in std noise as a function of phantom size')
                fig.append(Label('fig:noisereduction'))

        with doc.create(Subsection('Noise Texture')):

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'nps_images.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('2D noise power spectra made from repeat scans of water phantoms of a given diameter')
                fig.append(Label('fig:npsimages'))


            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'nps.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.5\linewidth'))
                fig.add_caption('Unnormalized noise power spectra across phantom sizes')

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'mean_nps.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('Change in mean NPS with phantom diameter')


        with doc.create(Subsection('Low Contrast Detectability Task Performance')):

            with doc.create(Subsection('AUC vs. Dose')):

                with doc.create(Figure(position='h!')) as fig:
                    image_filename = results_dir/'auc_v_dose_averagemm.png'
                    fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                    fig.add_caption('Average across phantom sizes')

                with doc.create(Figure(position='h!')) as fig:
                    for diameter in diameters:
                        image_filename = results_dir/f'auc_v_dose_{diameter:03d}mm.png'
                        with doc.create(SubFigure(
                                position='b',
                                width=NoEscape(r'0.3\linewidth'))) as subfig:

                            subfig.add_image(str(image_filename.absolute()),
                                                width=NoEscape(r'\linewidth'))
                            subfig.add_caption(f'{diameter} mm phantom')
                    fig.add_caption('Low contrast detectability AUC as a function of dose level measured in different phantom sizes (a) 112 mm diameter phantom, (b) a 151 mm phantom, and (c) a median sized-adult phantom at 292 mm.')

            doses = [25, 55, 100]

            with doc.create(Subsection('AUC vs. Diameter')):

                with doc.create(Figure(position='h!')) as fig:
                    image_filename = results_dir/'auc_v_diameter_averagedose.png'
                    fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                    fig.add_caption('Average across dose levels')

                with doc.create(Figure(position='h!')) as fig:
                    for dose in doses:
                        image_filename = results_dir/f'auc_v_diameter_{dose:03d}dose.png'
                        with doc.create(SubFigure(
                                position='b',
                                width=NoEscape(r'0.3\linewidth'))) as subfig:

                            subfig.add_image(str(image_filename.absolute()),
                                                width=NoEscape(r'\linewidth'))
                            subfig.add_caption(f'{dose} % dose')

                doc.append("Now let's consider the $\Delta AUC$ to see the potential task advantage of applying the denoiser")

                with doc.create(Figure(position='h!')) as fig:
                    image_filename = results_dir/'diffauc_v_diameter.png'
                    fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.5\linewidth'))
                    fig.add_caption('$\Delta AUC$ as a function of phantom diameter')

                doc.append("Now let's consider the $\Delta AUC$ for each HU insert")

                with doc.create(Figure(position='h!')) as fig:
                    image_filename = results_dir/'diffauc_v_diameter_hu.png'
                    fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.8\linewidth'))
                    fig.add_caption('$\Delta AUC$ as a function of phantom diameter and insert HU')
    
    doc.append(NoEscape(r'''\bibliographystyle{ieeetr}\n\bibliography{references}'''))

    doc.generate_pdf(results_dir/'summary', clean_tex=False)

if __name__ == '__main__':
    parser = ArgumentParser(description='Make Image Quality Summary Plots')
    parser.add_argument('results_directory', type=str, default="", help='directory containing results to be summarized')
    parser.add_argument('notes', nargs='?', type=str, default=None, help='list any additional experiment details to be included in the report')
    args = parser.parse_args()

    results_dir = args.results_directory
    main(results_dir, notes=args.notes)
