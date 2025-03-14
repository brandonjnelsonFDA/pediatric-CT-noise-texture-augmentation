# %%
import os
from pathlib import Path
from denoising.data import AugmentedDataset, AugmentedDataModule, MayoLDGCDataset, MayoLDGCDataModule, PediatricIQDataset
from dotenv import load_dotenv
import numpy as np
load_dotenv()

from PIL import Image
import pydot

import matplotlib.pyplot as plt
from notebooks.make_noise_patches import plot_representative_noise_patches


def prep_image_patches(output_dir, patch_size=64, group='adolescent', region='chest'):
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

    for x, y in aug_dl:
        break
    row = 0
    col = 0 
    plt.imshow(y[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'fd_patch.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.imshow(x[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'qd_patch.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.imshow(noise_patch[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'noise_patch.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.imshow(y[row, col] + noise_patch[row, col], cmap='gray', vmin=-250, vmax=350)
    plt.axis('off')
    plt.savefig(output_dir / 'aug_patch.png', dpi=300, bbox_inches='tight')
    plt.show()
    return output_dir


def make_traditional_dot_plot(img_dir, img_font=16):
    fname = 'fig1a_traditional'
    'http://magjac.com/graphviz-visual-editor/'
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
    fig_dir = Path(img_dir) / 'dot_charts'
    fig_dir.mkdir(exist_ok=True)
    graphs = pydot.graph_from_dot_data(dot_string)[0]
    output = fig_dir / f'{fname}.pdf'
    graphs.write_pdf(output)
    print(f'{output}')
    output = fig_dir / f'{fname}.png'
    graphs.write_png(output)
    print(output)

# %% Figure 1
def make_augmented_dot_plot(img_dir, img_font=15):
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

def main():
    img_dir = 'images'
    output_dir = prep_image_patches(img_dir)
    make_augmented_dot_plot(output_dir, img_font=15)
    make_traditional_dot_plot(output_dir, img_font=16)

if __name__ == '__main__':
    main()

    # noise_image_dict = dict()
    # subgroups = ['newborn', 'infant', 'child', 'adolescent', 'adult']
    # n_examples = 10
    # for group in subgroups:
    #     noise_aug = AugmentedDataset(MayoLDGCDataset, dict(root=os.environ['LDGC_PATH'], region='abdomen'),
    #                                  PediatricIQDataset, dict(root=os.environ['PEDIATRICIQ_PATH'],
    #                                                      phantom='uniform', subgroup=group), proportion=1)
    #     imgs = []
    #     for idx in range(n_examples):
    #         x, y = noise_aug[idx]
    #         imgs.append(x - y)
    #     noise_image_dict[group] = np.stack(imgs)

    # corner_patches, corners = plot_representative_noise_patches({k: v for k,v in noise_image_dict.items() if k in subgroups},
    #                                                             patch_size=patch_size)
    # noise_patch = corner_patches['newborn'][(110, 110)][0]