from argparse import ArgumentParser
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pydot
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from notebooks.make_noise_patches import plot_representative_noise_patches
from denoising.data import AugmentedDataset, AugmentedDataModule, MayoLDGCDataset, MayoLDGCDataModule, PediatricIQDataset
load_dotenv()

def prep_image_patches(output_dir, patch_size=64, group='adolescent', region='chest'):
    """
    This function prepares and saves the image patches for the flowchart.

    Parameters:
        output_dir (str): The directory to save the image patches.
        patch_size (int): The size of the image patches. Default is 64.
        group (str): The subgroup to use from the PediatricIQDataset. Default is 'adolescent'.
        region (str): The region to use from the MayoLDGCDataset. Default is 'chest'.

    Returns:
        output_dir (Path): The path to the directory where the image patches are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir.absolute()
    aug_dm = AugmentedDataModule(MayoLDGCDataset, dict(root=os.environ['LDGC_PATH'], region=region),
                                PediatricIQDataset, dict(root=os.environ['PEDIATRICIQ_PATH'],
                                phantom='uniform', subgroup=group), proportion=0, patch_size=patch_size, num_workers=1)
    aug_dm.setup('fit')
    aug_dl = aug_dm.train_dataloader()

    real_dm = MayoLDGCDataModule(os.environ['LDGC_PATH'], region=region, patch_size=patch_size, num_workers=1)
    real_dm.setup('fit')
    real_dl = real_dm.train_dataloader()

    for x, y in aug_dl:
        noise_patch = x - y
        break

    for x, y in real_dl:
        break

    row = 0
    col = 0
    plt.imshow(y[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'fd_patch.png', dpi=300, bbox_inches='tight')
    plt.imshow(x[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'qd_patch.png', dpi=300, bbox_inches='tight')
    plt.imshow(noise_patch[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'noise_patch.png', dpi=300, bbox_inches='tight')
    plt.imshow(y[row, col] + noise_patch[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'aug_patch.png', dpi=300, bbox_inches='tight')
    plt.close('all')  # close all figures to free memory
    return output_dir

def make_flowchart(img_dir, img_font=16, flowchart_type='traditional'):
    """
    This function creates a flowchart and saves it as a PDF and PNG.

    Parameters:
        img_dir (str): The directory containing the input images for the flowchart.
        img_font (int): The font size for the labels in the flowchart. Default is 16.
        flowchart_type (str): Type of flowchart to generate, either 'traditional' or 'augmented'. Default is 'traditional'.
    """
    if flowchart_type == 'traditional':
        fname = 'fig1a_traditional'
        dot_string = f"""digraph dot_image {{
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif"]
            edge [fontname="Helvetica,Arial,sans-serif"]
            layout=dot
            node [shape=box]; {{node [label=<<TABLE border="0" cellborder="0">
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font-2}">LD Adult Protocol</font></td></TR>
                                            <TR><TD width="100" height="100" fixedsize="true"><IMG SRC="{img_dir}/qd_patch.png" scale="true"/></TD></TR>
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font}">Training Input</font></td></TR>
                                            </TABLE>>] input}};
            node [shape=box]; {{node [label=<<TABLE border="0" cellborder="0">
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font-2}">FD Adult Protocol</font></td></TR>
                                            <TR><TD width="100" height="100" fixedsize="true"><IMG SRC="{img_dir}/fd_patch.png" scale="true"/></TD></TR>
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font}">Training Target</font></td></TR>
                                            </TABLE>>] target}};
            node [shape=box]; Prediction;

            node [shape=ellipse]; Model;
            node [shape=diamond,style=filled,color=lightgrey]; {{node [label="Loss Function"] loss}};

            input -> Model [len=1.00];

            Model -> Prediction [len=1.00];
            Prediction -> loss [len=1.00];

            target -> loss [len=1.00];
            loss -> Model [len=1.00];

            fontsize=24;
        }}"""
    else:  # 'augmented'
        fname = 'fig1b_augmented'
        dot_string = f"""digraph {{
            fontname="Helvetica,Arial,sans-serif"
            node [fontname="Helvetica,Arial,sans-serif"]
            edge [fontname="Helvetica,Arial,sans-serif"]
            layout=dot
            node [shape=box]; {{node [label=<<TABLE border="0" cellborder="0">
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font-2}">LD Adult Protocol</font></td></TR>
                                            <TR><TD width="100" height="100" fixedsize="true"><IMG SRC="{img_dir}/qd_patch.png" scale="true"/></TD></TR>
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font}">Training Input</font></td></TR>
                                            </TABLE>>] input}};
            node [shape=box]; {{node [label=<<TABLE border="0" cellborder="0">
                                        <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font-2}">FD Adult Protocol</font></td></TR>
                                    <TR><TD width="100" height="100" fixedsize="true"><IMG SRC="{img_dir}/fd_patch.png" scale="true"/></TD></TR>
                                    <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font}">Training Target</font></td></TR>
                                    </TABLE>>] target}};
            node [shape=box]; Prediction;
            node [shape=box]; {{node [label=<<TABLE border="0" cellborder="0">
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font-2}">Pediatric Protocol</font></td></TR>
                                            <TR><TD width="100" height="100" fixedsize="true"><IMG SRC="{img_dir}/noise_patch.png" scale="true"/></TD></TR>
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font}">Noise Patches</font></td></TR>
                                            </TABLE>>] patches}};
            node [shape=box]; {{node [label=<<TABLE border="0" cellborder="0">
                                            <TR><TD width="100" height="100" fixedsize="true"><IMG SRC="{img_dir}/aug_patch.png" scale="true"/></TD></TR>
                                            <TR><td width="100" height="10" fixedsize="true"><font point-size="{img_font}">Augmented Input</font></td></TR>
                                            </TABLE>>] augmented}};

            node [shape=ellipse]; Model;
            node [shape=diamond,style=filled,color=lightgrey]; {{node [label="Loss Function"] loss}};

            patches -> augmented;
            target ->  augmented [len=1.00];

            input -> Model  [label="1 - λ"];
            augmented -> Model  [label="λ"];

            Model -> Prediction [len=1.00];
            Prediction -> loss [len=1.00];
            target -> loss [len=1.00];
            loss -> Model [len=1.00];

            fontsize=24;
        }}"""

    fig_dir = Path(img_dir) / 'dot_charts'
    fig_dir.mkdir(exist_ok=True)
    graphs = pydot.graph_from_dot_data(dot_string)[0]
    output = fig_dir / f'{fname}.pdf'
    graphs.write_pdf(output)
    print(f'{output}')
    output = fig_dir / f'{fname}.png'
    graphs.write_png(output)
    print(output)

def main(img_dir = 'figures', flowchart_type='traditional', img_font=16):
    """
    This function is the main function that calls the other functions to create the flowcharts.

    Parameters:
        img_dir (str): The directory to save the images and flowcharts. Default is 'figures'.
        flowchart_type (str): Type of flowchart to generate, either 'traditional' or 'augmented'. Default is 'traditional'.
        img_font (int): The font size for the labels in the flowchart. Default is 16.
    """
    output_dir = prep_image_patches(img_dir)
    make_flowchart(output_dir, img_font=img_font, flowchart_type=flowchart_type)

if __name__ == '__main__':
    parser = ArgumentParser(description='This script generates a flow chart for Figure 1 of the denoising project.')
    parser.add_argument('--output', '-o', type=str, default='figures', help='The directory to save the images and flowcharts. Default is "figures".')
    parser.add_argument('--type', '-t', type=str, choices=['traditional', 'augmented'], default='traditional', help='The type of flow chart to generate. Options are "traditional" and "augmented". Default is "traditional".')
    parser.add_argument('--font-size', '-f', type=int, default=16, help='The font size for the labels in the flowchart. Default is 16.')
    args = parser.parse_args()
    main(args.output, args.type, args.font_size)
