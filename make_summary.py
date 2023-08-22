#%%
from pylatex import Document, Section, Subsection, Figure, SubFigure, NoEscape, Command
from pylatex.labelref import Label

from pathlib import Path
from argparse import ArgumentParser

# %%
def main(results_dir, notes=None):
    results_dir = Path(results_dir)
    geometry_options = {"tmargin": "1in","bmargin":"1in", "lmargin": "1in", "rmargin": "1in"}
    doc = Document(geometry_options=geometry_options)

    doc.preamble.append(Command('title', 'Size-Based Noise Data Augmentation to Improve Generalizability of Deep Learning Denoising in Pediatric CT'))
    doc.preamble.append(Command('author', 'Brandon J. Nelson'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    diameters = [112, 151, 292]

    if notes:
        with doc.create(Section('Experiment Details')):
            doc.append(notes)

    with doc.create(Section('Methods')):

        doc.append(NoEscape(r'Water cylinders of different sizes scanned with fitting FOVs and reconstructed with different methods (Figure \ref{fig:images}): standard filtered backprojection (FBP), FBP followed by a simple denoiser Simple CNN with MSE, and the same denoiser trained with the proposed data augmentation scheme'))

        with doc.create(Figure(position='h!')) as fig:
            image_filename = results_dir/'images.png'
            fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
            fig.add_caption('Water cylinders of different sizes scanned with fitting FOVs and reconstructed with different methods.')
            fig.append(Label('fig:images'))

        with doc.create(Figure(position='h!')) as fig:
            image_filename = results_dir/'noise_images.png'
            fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
            fig.add_caption('Sample noise images made from repeat scans of water phantoms of a given diameter')
            fig.append(Label('fig:noise images'))
    
        doc.append(NoEscape(r"Repeat water scans from (Figure \ref{fig:images}) were subtracting in different combinations yielding 2000 noise images per setting yielding the noise images in (Figure \ref{fig:noise images})"))

    with doc.create(Section('Noise Reduction')):
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

    with doc.create(Section('Noise Texture')):

        with doc.create(Figure(position='h!')) as fig:
            image_filename = results_dir/'nps_images.png'
            fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
            fig.add_caption('2D noise power spectra made from repeat scans of water phantoms of a given diameter')

        with doc.create(Figure(position='h!')) as fig:
            image_filename = results_dir/'nps.png'
            fig.add_image(str(image_filename.absolute()), width=NoEscape(r'0.5\linewidth'))
            fig.add_caption('Unnormalized noise power spectra across phantom sizes')

        with doc.create(Figure(position='h!')) as fig:
            image_filename = results_dir/'mean_nps.png'
            fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
            fig.add_caption('Change in mean NPS with phantom diameter')


    with doc.create(Section('Low Contrast Detectability Task Performance')):

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
                fig.add_image(str(image_filename.absolute()), width=NoEscape(r'\linewidth'))
                fig.add_caption('$\Delta AUC$ as a function of phantom diameter and insert HU')

    doc.generate_pdf(results_dir/'summary', clean_tex=True)

if __name__ == '__main__':
    parser = ArgumentParser(description='Make Image Quality Summary Plots')
    parser.add_argument('results_directory', type=str, default="", help='directory containing results to be summarized')
    parser.add_argument('notes', nargs='?', type=str, default=None, help='list any additional experiment details to be included in the report')
    args = parser.parse_args()

    results_dir = args.results_directory
    main(results_dir, notes=args.notes)
