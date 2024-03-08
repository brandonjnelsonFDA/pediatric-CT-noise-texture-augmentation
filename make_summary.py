#%%
from pylatex import Document, Section, Subsection, Figure, SubFigure, NoEscape, Command, Table
from pylatex.labelref import Label

from pathlib import Path
from argparse import ArgumentParser

# %%
def main(args):
    results_dir = Path(args.results_directory)
    notes=args.notes
    patch_size = args.patch_size
    geometry_options = {"tmargin":"1in", "bmargin":"1in", "lmargin":"1in", "rmargin":"1in"}
    doc = Document(geometry_options=geometry_options)

    doc.preamble.append(Command('title', 'Noise Texture Augmentation to Improve Generalizability of Deep Learning Denoising in Pediatric CT'))
    doc.preamble.append(Command('author', 'Brandon J. Nelson, Prabhat Kc, Andreu Badal, Lu Jiang, Rongping Zeng'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.preamble.append(NoEscape(r'\usepackage{booktabs}'))
    doc.append(NoEscape(r'\maketitle'))

    diameters = [112, 151, 292]

    if notes:
        with doc.create(Section('Experiment Details')):
            doc.append(notes)

    with doc.create(Section('Abstract')):
        doc.append('''
The goal of this work is to leverage data augmentation, a deep learning model training technique for enhancing limited training datasets, as a means to improve deep learning CT denoising to patients of sizes outside their training distribution.''')

    with doc.create(Section('Introduction')):
        doc.append(NoEscape(r'''
Deep learning image reconstruction and denoising have been shown to be a viable option for reducing noise in CT imaging, enabling dose reductions on par with or potentially greater than the previous state of the art model-based iterative reconstruction. In addition deep learning denoising, requires less computation at inference time and can better preserve noise texture relative to filtered back projection (FBP), a feature generally favored by radiologists and known to affect low contrast detectability (**cite**).\n
        
However, a key limitation of deep learning techniques more broadly is their limited ability to generalize to data characteristically different than they were trained with. Specifically for deep learning CT denoising models have been shown to be particularly sensitive to changes in noise texture of the input image due to differences in reconstruction kernels and reconstructed fields of view (FOV).\cite{huberEvaluatingConvolutionalNeural2021} Zeng et al., 2022 demonstrated deep denoisers to behave like band-pass filters, removing noise from spatial frequencies included in the training set, while ignoring spatial frequencies outside of the training distribution.\cite{zengPerformanceDeepLearningbased2022} A negative consequence of this band-pass behavior is the potential for reduced performance in imaging pediatric patients. This reduced performance was shown to correlate with the reduced body-size and thus smaller reconstructed field of view (FOV) associated with pediatric protocols. \cite{nelsonPediatricSpecificEvaluationsDeep2023} This poor performance raises health equity concerns as it could limit access for pediatric patients to the latest deep learning enabled medical advancements. Pediatric patients are under represented in radiological imaging, making up only 5\% of scans despite representing 20\% of the US population,\cite{smith-bindmanTrendsUseMedical2019} thus large pediatric datasets are not available to develop deep learning models for pediatric patients. **(can also cite Marla's ACR peds white paper)**

In other deep learning applications where the training dataset is small and not representative of the general population, data augmentation is one strategy to effectively enlarge the training dataset and potentially improve a model's generalizability. Data augmentation works by assuming that the variability of certain features in the general population can be modeled and incorporated into model training, for example adding some random horizontal flips, rotations or intensity changes to the training images can be effective if such variability is expected in the general population of images. The hypothesis of this work is that noise textures from pediatric protocol CT images missing from adult protocol CT image training datasets can be supplemented into deep learning denoiser model training as a data augmentation to improve denoising performance pediatric CT. Thus the goal of this work is to leverage data augmentation, a deep learning model training technique for enhancing limited training datasets, as a means to improve deep learning CT denoising to patients of sizes outside their training distribution. This is done by extracting noise textures extracted from scans of phantoms representative of pediatric sizes and FOVs to augment the adult training data. The result is a denoising model that generalizes better to smaller patients, saving time, resources, and radiation exposure compared to compiling large datasets with these patients, which is generally not feasible.
'''))

    with doc.create(Section('Methods')):
        doc.append(NoEscape(r'''
Figure \ref{fig:schematic} compares traditional model training (Figure \ref{fig:schematic}a) with our proposed noise texture augmented model training Figure \ref{fig:schematic}b. Traditional deep learning denoising models are trained using low dose training inputs and high dose training targets where the model processes the noisy input, attempting to remove noise in its prediction. This prediction is compared to the high dose training target using the loss function and the model is then updated to minimize this loss function and the process repeats. As these training inputs and targets generally are from adult CT image datasets, this approach works well in adults of similar size as in the training, but have been shown to perform worse in pediatric patients who are smaller than the adults in the training set.\cite{nelsonPediatricSpecificEvaluationsDeep2023} In our proposed noise texture augmented training Figure \ref{fig:schematic}b noise patches are generated from simulated CT scans of phantoms representative of different pediatric waist diameters ranging from newborn to adolescent using body fitting FOVs. These patches have distinct noise textures and are combined with the high dose training target images to make a new augmented input estimating a low dose image from a smaller FOV pediatric patient. These augmented inputs are used together the original low dose training inputs making up a proportion $\lambda$ of the total training data. This proportion $\lambda$, which controls the magnitude of augmentation along with the characteristics of the generated noise patches all contribute to the augmented training model performance. 

                            (**TODO how do noise textures look in pediatric only data**)

To assess whether the proposed augmentation technique (Figure \ref{fig:schematic}b) improved generalizability in smaller patients compared to the traditionally trained model (Figure \ref{fig:schematic}a), performance was assessed as the magnitude of noise reduction, noise texture preservation, image sharpness, and low contrast detectability using model observers.
'''))
        with doc.create(Figure(position='h!')) as fig:
            with doc.create(SubFigure(
                                position='b',
                                width=NoEscape(r'0.4\linewidth'))) as subfig:
                image_filename = results_dir/'standard_training_schematic.png'
                subfig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.9\linewidth'))
                subfig.add_caption('Traditional Model Training')
            with doc.create(SubFigure(
                                position='b',
                                width=NoEscape(r'0.5\linewidth'))) as subfig:
                image_filename = results_dir/'augmented_training_schematic.png'
                subfig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                subfig.add_caption('Noise Texture Augmented Model Training')
            fig.add_caption(NoEscape(r'Schematic diagram of proposed data augmentation. (a) Standard model training where training input patches are given to a model to make a prediction that is compared against the training target using the loss function used to update the model. (b) In the augmented training noise patches are added to the low noise training target to make a new augmented input. While training, the proportion $\lambda$ of the training data is from the augmented inputs while the remaining $1-\lambda$ is from the original low dose training inputs. This augmentation parameter is an additional training hyperparameter introduced by this method.')) #augmentation is a form of regularization
            fig.append(Label('fig:schematic'))

        with doc.create(Subsection('Making Noise Texture Patches')):
            doc.append(NoEscape(
r'''
Water cylinders of different sizes were numerically simulated with CT projection data simulated using the Michigan Image Reconstruction Toolbox (MIRT). The acquisition parameters were modeled after the Siemens Sensation scanner with noise texture, sharpness, mA and kVp matching those used in the Mayo Clinic's Low Dose Grand Challenge Dataset.\cite{mccolloughLowdoseCTDetection2017} CT images were reconstructed from this projection data using fitting FOVs equal to 110\% the cylinder diameter as shown in Figure \ref{fig:methods}a. CTDIvol in the simulations was scaled by adjusting mA in the simulations such that the noise, measured as the standard deviation of voxel intensities in a region of interest, were approximately constant for all phantom sizes.\cite{nelsonPediatricSpecificEvaluationsDeep2023}
'''))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'noise_texture_fbp.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                fig.add_caption(f'Creating noise patches of varying texture for data augmentation. (a) Water phantoms of varying diameters (112, 185, and 216 mm are shown) are virtually scanned, [mean, standard deviation] are shown for different {patch_size}x{patch_size} pixel patches from different regions of the image. Taking the difference of multiple repeat scans with different instances of projection Poisson noise yields the noise only images (b). (c) patches taken from different regions around the noise images show different noise orientations. Noise grain size also decreases with increasing phantom size. (d) 2D Noise power spectra illustrate different orientations and spatial frequencies of noise between patches taken from different regions and phantom sizes.')
                fig.append(Label('fig:methods'))

            doc.append(NoEscape(r'''
Noise only images from each size cylinder phantom image were then made by taking the difference of repeat scans with different projection noise instances (Figure \ref{fig:methods}b). Note that the measured noise from central and peripheral regions is approximately $\sqrt{2\sigma^2}$, where $\sigma$ is the measured noise from the reconstructed images in Figure \ref{fig:methods}b because we defined noise as the standard deviation, rather than the variance. Patches were then selected from random locations across these noise only images (Figure \ref{fig:methods}c). The noise only images are split into patches since most denoising models are trained on image patches rather than whole image slices (**cite**). The matrix size of these random patches is set to match the matrix size of the training set image patches. By selecting random locations from the noise images, these noise patches contain different orientations of noise, shown by the different orientation of NPS images taken from the center, top and left of the noise images (Figure \ref{fig:methods}d). The noise patches from different sized phantoms also contain noise of varying grain size. Noise patches from the smaller diameter phantoms have larger noise grains and thus their noise power spectra are predominantly lower frequency, while the smaller noise grain patches from large FOV phantom scans are higher frequency (Figure \ref{fig:methods}d).'''))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'trainingnoise.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.6\linewidth'))
                fig.add_caption('Comparison of noise textures found in training dataset compared to those to be added via data augmentation. (a) Example training inputs. (b) Noise textures estimated by taking the difference between training inputs and targets. (c) The averaged noise power spectra (NPS) across noise textures found in training. (d) Averaged NPS across generated noise textures from each diameter phantom to be used for augmented training.')
                fig.append(Label('fig:trainingnoise'))

            doc.append(NoEscape(r'''
Compared to noise images from the training set (Figure \ref{fig:trainingnoise}b), found by taking the difference between training inputs (Figure \ref{fig:trainingnoise}a) and training targets, the phantom simulated noise patches encompass a wider range of noise spatial frequencies than encountered in the adult-only training set as shown by comparing the averaged training noise NPS in Figure \ref{fig:trainingnoise}c with that of the generated noise patches to be used for augmentation shown in Figure \ref{fig:trainingnoise}d.\n
'''))

        with doc.create(Subsection('Denoising Model Training with Augmentation')):
            doc.append(
'''
The goal of our proposed size-based noise data augmentation is to incorporate this diversity of noise textures into the model training loop to improve the model generalizability to remove noise from a wider range of noise textures as would be seen in smaller patients and pediatric patients. This is illustrated in Figure \ref{fig:schematic}, where every X percent of training examples (**include equation with relevant parameters to describe the augmentation**), the usual low dose input, high dose training pair is replaced with a new augmented training pair. The new input is the high dose input with an added random noise patch where the ta 
''')

        with doc.create(Subsection('Size Generalization Evaluations')):
            doc.append(NoEscape(
r'''
To assess whether the data augmentation was able to enhance the generalizability model to different sized phantoms, evaluations of noise reduction, sharpness, noise texture, and low contrast detectability were performed.\n
'''))
    
    with doc.create(Section('Results')):

        with doc.create(Subsection('Noise Reduction')):
            doc.append(NoEscape(r"Our first assessment compares noise reduction performance measured across phantoms effective diameter ranging from newborn pediatric patients through adult \ref{fig:anthro_patient_denoising_comp}. By using automatic exposure control with a reference FBP noise level of 48 HU for the quarter dose acquisitions, all FBP reconstructions in Fig. \ref{fig:anthro_patient_denoising_comp} have consistent noise magnitude measurements taken from liver ROIs in all three effective diameters shown (11.2, 16.9, and 34.2 cm). Following deep denoising with a RED-CNN model, a correlation with size appears, as the measured noise magnitude in liver ROIs decreases with increasing effective diameter, from 24 HU in the newborn 11.2 cm diameter XCAT down to 10 HU in the adult 34.2 cm diameter XCAT. However, this size dependence is no longer present in augmented training RED-CNN, trained on the same adult CT dataset. After denoising from the RED-CNN augmented model, all quarter dose FBP input images shown went from a measured noise magnitude of around 48 HU in the liver to 12 HU, independent of size and FOV."))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'anthro_montage.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('CT simulations of XCAT anthropomorphic phantoms of different effective diameter reconstructed with Filtered back projection (FBP) followed by denoising with different deep learning denoising models. Region of interest (ROI) measurements taken from the liver report pixel mean and standard deviation with the ROI.')
                fig.append(Label('fig:anthro_patient_denoising_comp'))

            doc.append(NoEscape(r"These trends are more clear when plotting measures of noise and noise reduction as a function of phantom effective diameter (Figure \ref{fig:noisereduction}). Noise is reported in two ways: 1) as noise magnitude - measured as pixel standard deviation from a circular liver ROI with diameter 20\% the patient effective diameter, and 2) as the root mean squared error (RMSE) with the ground truth phantom image, free of noise, blurring, or aliasing effects. In agreement with Fig. \ref{fig:anthro_patient_denoising_comp}, noise magnitude - measured as noise std in liver ROIs - is consistent across phantom size for both FBP and the augmented RED-CNN compared to the original RED-CNN model which is biased toward higher noise std in small effective diameter phantoms. Consistent with Nelson et al., 2023 \cite{nelsonPediatricSpecificEvaluationsDeep2023}, noise measurements of deep denoised images made in uniform water phantoms correlate with anthropomorphic phantom measurements, though anthropomorphic noise std measurements are generally lower as these anthropomorphic phantoms are more similar to the patient training data than the uniform water cylinder IQ phantoms used for bench testing."))

            doc.append(NoEscape(r"Another observation from is that data augmentation act as a regularizer, yielding more consistent, generalizable performance across patient size, but at the expense of reduced maximal performance in some larger adult patients. For example, when averaged across all effective diameters augmented RED-CNN had an average noise std reduction of 70\% compared to 61\% for the base RED-CNN, but among effective diameters greater than 30 cm - corresponding to larger adult patients - the augmented RED-CNN under-performed the base RED-CNN with a noise std reduction of 73\% compared to 78\% for base RED-CNN. These additional findings are summarized in Table XX. which also shows that the performance gap in adult patients narrows when considering RMSE."))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'noise_reduction.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                fig.add_caption('Reduction in std noise as a function of phantom size')
                fig.append(Label('fig:noisereduction'))

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'subgroup_denoising_comparison.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.7\linewidth'))
                fig.add_caption('Comparison of noise reduction performance across size-based subgroups')
                fig.append(Label('fig:subgroup_noisereduction'))

            doc.append(NoEscape(r"""
% Please add the following required packages to your document preamble:
% \usepackage{booktabs}
\begin{table}[]
\begin{tabular}{@{}lll@{}}
\toprule
Subgroup & Age Range & Waist Diameter Range \\ \midrule
newborn & $\leq$ 1 mo & $\leq$ 11.5 cm \\
infant & \textgreater 1 mo \& $\leq$ 2 yrs & \textgreater 11.5 \& $\leq$ 16.8 cm \\
child & \textgreater 2 yrs \& $\leq$ 12 yrs & \textgreater 16.8 \& $\leq$ 23.2 cm \\
adolescent & \textgreater 12 yrs \& \textless 21 yrs & \textgreater 23.2 \& \textless 34 cm \\
adult & $\geq$ 22 yrs & $\geq$ 34 cm \\ \bottomrule
\end{tabular}
\caption{my table}
\label{tab:my-table}
\end{table}
"""))

        with doc.create(Subsection('Noise Texture')):

            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'nps_images.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('2D noise power spectra made from repeat scans of water phantoms of a given diameter')
                fig.append(Label('fig:npsimages'))


            with doc.create(Figure(position='h!')) as fig:
                image_filename = results_dir/'nps.png'
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.5\linewidth'))
                fig.add_caption('Unnormalized noise power spectra across phantom size')

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
                    fig.append(Label('fig:taskperformance'))

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
    with doc.create(Section('Discussion')):
        doc.append(NoEscape(r'These findings, in particular the size dependence of the RED-CNN denoiser noise reduction (Fig. \ref{fig:noisereduction}) and task performance (Fig. \ref{fig:taskperformance}) are consistent with our previous findings (Nelson et al., 2023 \cite{nelsonPediatricSpecificEvaluationsDeep2023})) but have now been expanded upon using a data augmentation-based intervention to improve pediatric performance of deep learning denoising models without requiring further pediatric training data. In this work, we have showed that by incorporating a simple data augmentation scheme (Fig. \ref{fig:schematic}b) using noise textures extracted from pediatric protocol phantom scans (Fig. \ref{fig:methods}). The resulting models have better noise reduction in terms of noise magnitude and RMSE (Fig. \ref{fig:noisereduction}), improved task-based performance (Fig. \ref{fig:taskperformance}), and noise textures with mean spatial frequencies closer to FBP (Fig. XX)'))

        doc.append(NoEscape(r'This study has limitations. Due to limited availability in pediatric patient data, pediatric evaluations were limited to pediatric XCAT phantoms and protocols and simulation image quality phantoms.'))

        doc.append(NoEscape(r'Future investigations include exploring different weighting schemes, currently noise textures from different sized phantoms were uniformly incorporated at a constant frequency of $\lambda$ (Fig. \ref{fig:schematic}b), but redefining $\lambda$ as a vector with different weighting values for each phantom size could allow for more fine tuning of denoising performance and texture across patient sizes to counteract known under-representation bias in a patient population or achieve more desirable noise texture characteristics by over-weighting specific frequencies.'))
    
    with doc.create(Section('Conclusions')):
        doc.append('Data augmentation using noise textures extracted from pediatric-sized phantom scans were shown to yield improved denoising performance across all patient sizes.')

    with doc.create(Section('References')):    
        doc.append(NoEscape(r'''\bibliographystyle{ieeetr}\n\bibliography{references.bib}'''))

    doc.generate_pdf(results_dir/'summary', clean_tex=False)

if __name__ == '__main__':
    parser = ArgumentParser(description='Make Image Quality Summary Plots')
    parser.add_argument('results_directory', type=str, default="", help='directory containing results to be summarized')
    parser.add_argument('notes', nargs='?', type=str, default=None, help='list any additional experiment details to be included in the report')
    parser.add_argument('--patch_size', type=int, default=30, help='side length of square patches to be extracted, e.g. patch_size=30 yields 30x30 patches')
    args = parser.parse_args()
    main(args)

# %%