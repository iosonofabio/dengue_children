# vim: fdm=indent
'''
author:     Fabio Zanini
date:       16/03/20
content:    Cell type 4 kids samples
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
import loompy
from scipy.io import mmread

import matplotlib.pyplot as plt
import seaborn as sns


os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet import Dataset, CountsTable, FeatureSheet, SampleSheet, concatenate


def exclude_hla(features):
    prefixes = [
            'IGHV', 'IGHJ', 'IGKV', 'IGKJ', 'IGLV', 'IGLJ',
            'TRAV', 'TRAJ', 'TRBV', 'TRBJ',
            ]
    ind = features.str.startswith(prefixes[0])
    for pfx in prefixes[1:]:
        ind |= features.str.startswith(pfx)
    ind = ~ind
    return features[ind]


if __name__ == '__main__':

    print('Load data for 4kids')
    ds = Dataset(
        dataset={
            'path': '../../data/datasets/20200313_4kids/20200313_4kids.loom',
            'index_samples': 'CellID',
            'index_features': 'GeneName',
            'bit_precision': 32,
            })

    print('Feature selection')
    features = ds.feature_selection.overdispersed_within_groups('Sample', inplace=False)
    features = exclude_hla(features)

    if 'umap_1' not in ds.samplesheet.columns:
        dsf = ds.query_features_by_name(features)

        print('PCA')
        dsc = dsf.dimensionality.pca(n_dims=30, robust=False, return_dataset='samples')

        print('UMAP (tSNE is too slow)')
        vs = dsc.dimensionality.umap()
    else:
        vs = ds.samplesheet[['umap_1', 'umap_2']].rename(
                columns={'uamp_1': 'dimension 1', 'umap_2': 'dimension 2'},
                )

    if True:
        print('Plot a few marker genes')
        plt.ioff()
        fig, axs = plt.subplots(5, 10, figsize=(20, 10), sharex=True, sharey=True)
        axs = axs.ravel()
        fig2, ax = plt.subplots()
        axs = list(axs) + [ax]
        figs = [fig, fig2]
        marker_genes = [
                ('PTPRC', 'CD45'),
                'CD3E',
                'CD19',
                'MS4A1',
                'IGHM',
                'IGHD',
                'GZMA',
                'CD14',
                'TYROBP',
                #'PTPRS',
                'CD163',
                'FCGR3A',
                ]
        markers = [
                #('number_of_genes_1plusreads', 'n genes'),
                'Sex',
                #'cellType',
                'Condition',
                'Sample',
                ]
        marker_genes += [x for x in features if (not x.startswith('ERCC')) and (x not in ['ACTB', 'MALAT1'])][:len(axs) - len(marker_genes) - len(markers)]
        markers = marker_genes + markers
        mgs = [x if isinstance(x, str) else x[0] for x in marker_genes]
        for ipl, (gene, ax) in enumerate(zip(markers, axs)):
            print('Plotting gene {:} of {:}'.format(ipl+1, len(markers)))
            if isinstance(gene, str):
                gene, title = gene, gene
            else:
                gene, title = gene
            if gene == 'tissue':
                cmap = sns.color_palette('husl', n_colors=11)
            elif gene == 'Condition':
                cmap = {'control': 'steelblue', 'dengue': 'tomato'}
            elif gene == 'Sex':
                cmap = {'M': 'dodgerblue', 'F': 'deeppink'}
            else:
                cmap = 'viridis'
            ds.plot.scatter_reduced_samples(
                    vs,
                    ax=ax,
                    s=10,
                    alpha=0.05 + 0.2 * (gene in marker_genes),
                    color_by=gene,
                    color_log=(gene in mgs + ['number_of_genes_1plusreads']),
                    cmap=cmap,
                    )
            ax.grid(False)
            ax.set_title(title)

            if gene in ('Sample', 'Sex', 'Condition'):
                fig.tight_layout(rect=(0, 0, 0.87, 1))

                import matplotlib.lines as mlines
                d = ax._singlet_cmap
                handles = []
                labels = []
                for key, color in d.items():
                    h = mlines.Line2D(
                        [], [], color=color, marker='o', lw=0,
                        markersize=5,
                        )
                    handles.append(h)
                    labels.append(key.upper())
                ax.legend(
                        handles,
                        labels,
                        loc='lower right',
                        fontsize=6 + 3 * (gene in ['Sample']),
                        ncol=1,
                        #bbox_to_anchor=(0.99, 0.07),
                        #bbox_transform=fig.transFigure,
                        )
        for fig in figs:
            fig.tight_layout()
        #figs[0].savefig('../../figures/4kids_many_umaps.png')
        #figs[1].savefig('../../figures/4kids_umap_sample.png')

    plt.ion()
    plt.show()
