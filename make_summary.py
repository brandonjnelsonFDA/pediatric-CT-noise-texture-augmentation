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

    doc.preamble.append(Command('title', 'Size-Based Noise Data Augmentation to Improve Generalizability of Deep Learning Denoising in Pediatric CT'))
    doc.preamble.append(Command('author', 'Brandon J. Nelson'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    diameters = [112, 151, 292]

    if notes:
        with doc.create(Section('Experiment Details')):
            doc.append(notes)

    with doc.create(Section('Introducton')):
        doc.append(NoEscape(r'''
Deep learning image reconstruction and denoising has been shown to be a viable option for reducing noise in CT imaging, enabling dose reductions on par with or potentially greater than the previous state of the art, model-based iterative reconstruction. In addition deep learning denoising, requires less computation at inference time and can better preserve noise texture relative to filtered backprojection (FBP), a feature generally favored by radiologists and know to affect low contrast detectability (**cite**).\n
        
However, a key limitation of deep learning techniques, is their limited ability to generalize to data characteristically different from their training data. Prior studies have shown such deep learning CT denoising models are particularly sensitive to changes in noise texture of  the input image prior to denoising. These changes in noise texture can come from the input image being reconstructed with a different reconstruction kernel\cite{zengPerformanceDeepLearningbased2022} and reconstructed field of view (FOV).\cite{huberEvaluatingConvolutionalNeural2021} Such changes in noise texture are common when imaging different size patients, particularly in abdominal imaging, where FOV is routinely adapted to fit the patient. This effect was recently shown to have substantial influence when considering using deep learning denoising models trained on adult patiets but applied to pediatric patients. Given that pediatric patients can be considerably smaller than adult patient, it was shown that the smaller associated FOV results in reduction in denoising performance that progressively worsened as patient size and FOV decreased relative to the training distribution.\cite{nelsonPediatricSpecificEvaluationsDeep2023}\n
        
This demonstration that adult-trained deep learning models do not generalize to pediatric patient raises health equity concerns as it could limit access for pediatric patients to the latest and greatest medical advancements made available with deep learning. Pediatric patients are under represented in radiological imaging, making up only 5\% of scans despite representing 20\% of the US population. As a result deep learning models are generally not trained with pediatric data. The goal of this work is to leverage data augmentation, a deep learning model training technique for enhancing limited training datasets, as a means to improve deep learning CT denoising to patients of sizes outside of the training distribution. This is done by extracting noise textures extracted from scans of phantoms representative of pediatric sizes and FOVs to augment the adult traininging data. The result is denoising model that generalizes better to smaller patients, saving time, resources, and radiation exposure compared to compiling large datasets with these patients, which is generally not feasible.
'''))

    with doc.create(Section('Methods')):

        with doc.create(Subsection('Noise Texture Augmentation')):
            doc.append(NoEscape(
r'''
Water cylinders of different sizes were numerically simulated and with CT projection data simulated using the Michigan Image Reconstruction Toolbox (MIRT). The acquisition parameters were modeled after the Siemens Sensation scanner with noise texture, sharpness, mA and kVp matching those used in the Mayo Clinic's Low Dose Grand Challenge Dataset (**cite**). CT images were then reconstructed from this projection data using fitting FOVs equal to 110\% the cylinder diameter shown in Figure \ref{fig:images}a (Make a methods version of this figure showing just FBP (part a), the noise difference images (part b), and noise patches (part c) and NPS profiles (part c)).\n
'''))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'images.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('Water cylinders of different sizes scanned with fitting FOVs and reconstructed with different methods.')
                fig.append(Label('fig:images'))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'noise_images.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('Sample noise images made from repeat scans of water phantoms of a given diameter')
                fig.append(Label('fig:noiseimages'))
            doc.append(NoEscape(r'''
Noise only images from each size cylinder phantom image were then made by taking the dfference of all paired combinations (\ref{fig:images}b). Patches were then selected from random locations across these noise only images (\ref{fig:images}c). The noise only images are split into patches since most denoising models are trained on image patches rather than whole image slices (**cite**). The matrix size these random patches is set to match the matrix size of the training set image patches. By selecting random locations, these noise patches contain different orientations of noise, and noise patches from different sized phantoms contain noise of varying grain size ((\ref{fig:images}d) just the random image patches from different corners). This noise grain size can be quantified by the noise power spectra (NPS) where the larger noise grain from the smaller FOV phantom scans are predominantly lower frequency, while the smaller noise grain patches from large FOV phantom scans are higher frequency (\ref{fig:npsimages}a). Compared to noise images from the training set, found by taking the difference between training inputs and training targets, these phantom simulated noise patches encompasss a wider range of noise spatial frequencies than encountered in the adult-only training set.\n

The goal of our propose size-based noise data augmentation is to incorporate this diversity of noise textures into the model training loop to improve the model generalizability to remove noise from a wider range of noise textures as would be seen in smaller patients and pediatric patients. This is illustrated in diagram X, where every X percent of training examples, the usual low dose input, high dose training pair is replaced with a new augmented training pair. The new input is the high dose input with an added random noise patch where the ta 
            '''))
            doc.append(NoEscape(r"Repeat water scans from (Figure \ref{fig:images}) were subtracting in different combinations yielding 2000 noise images per setting yielding the noise images in (Figure \ref{fig:noiseimages})"))
    
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
