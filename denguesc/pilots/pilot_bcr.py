# vim: fdm=indent
'''
author:     Fabio Zanini
date:       20/11/20
content:    Have a look at the BCR seuences.
'''
import os
import sys
import numpy as np
import pandas as pd
import anndata

import matplotlib.pyplot as plt
import seaborn as sns




pname_conversion = {
        '1-002': '1_002_01',
        '6_023_01': '6_023_01',
        '6-001': '6_001_01',
        '1-075': '1_075_01',
        '5-044': '5_044_01',
        '3_047_01': '3_047_01',
        '5-041': '5_041_01',
        '1-140': '1_140_01',
        '5_030': '5_030_01',
        '6-020': '6_020_01',
        '3_037_01': '3_037_01',
        '3_012_01': '3_012_01',
        '6-025': '6_025_01',
        '1-144': '1_144_01',
        '5-154': '5_154_01',
        '5-089': '5_089_01',
        '5_193': '5_193_01',
        '1_019_01': '1_019_01',
        '6-028': '6_028_01',
}


condd = {
    '1_002_01': 'S_dengue',
    '1_019_01': 'dengue',
    '1_075_01': 'S_dengue',
    '1_140_01': 'S_dengue',
    '1_144_01': 'S_dengue',
    '3_012_01': 'Healthy',
    '3_037_01': 'Healthy',
    '3_047_01': 'Healthy',
    '3_074_01': 'Healthy',
    '5_030_01': 'S_dengue',
    '5_041_01': 'S_dengue',
    '5_044_01': 'S_dengue',
    '5_089_01': 'DWS',
    '5_154_01': 'dengue',
    '5_193_01': 'S_dengue',
    '6_001_01': 'dengue',
    '6_020_01': 'DWS',
    '6_023_01': 'dengue',
    '6_025_01': 'DWS',
    '6_028_01': 'DWS',
}



if __name__ == '__main__':

    fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
    adata = anndata.read_h5ad(fn_h5ad)
    adata = adata[adata.obs['platform'] == '10X']
    adata.obs['barcode'] = ['-'.join(x.split('-')[:-1]) for x in adata.obs_names]
    adata.obs['CellID'] = adata.obs['barcode'] + '-' + adata.obs['ID'].astype(str)

    fn = '../../data/datasets/20200809_20kids/20200810_20kids_vdj_all_contig_annotations.csv.gz'
    df = pd.read_csv(fn)
    df['patient_sample'] = df.patient_sample.map(pname_conversion)
    df['CellID'] = df['barcode'] + '-' + df['patient_sample']
    df['Condition'] = df['patient_sample'].map(condd)
    df['VJ combo'] = df['v_gene'] + ' ' + df['j_gene']

    print('Restrict to BCRs')
    chain_types = ['IGH', 'IGK', 'IGL']
    dfb = df.loc[df['chain'].isin(chain_types)]

    print('Restrict to is_cell')
    dfb = dfb.loc[dfb['is_cell']]

    print('Split into IGH, IGK, IGL')
    dfd = {x: dfb.loc[dfb['chain'] == x] for x in chain_types}

    print('Plot ranks')
    conditions = [
            'Healthy', 'dengue', #'DWS',
            'S_dengue']
    gene_types = ['v_gene', 'j_gene', 'VJ combo', 'raw_clonotype_id']
    cmap = {'Healthy': 'royalblue', 'dengue': 'orange', 'DWS': 'tomato', 'S_dengue': 'darkred'}
    nsub = 2000
    fig, axs = plt.subplots(4, 3, figsize=(7, 8))
    for ir, (axr, gt) in enumerate(zip(axs, gene_types)):
        for ic, (ax, ct) in enumerate(zip(axr, chain_types)):
            ymax = 0
            for cond in conditions:
                tmp = dfd[ct]
                tmp = tmp.loc[tmp['Condition'] == cond]

                # Subsample 1,000 lines
                idx = np.arange(len(tmp))
                np.random.shuffle(idx)
                idx = idx[:nsub]
                tmp = tmp.iloc[idx]

                y = tmp[gt].value_counts()
                ymax = max(ymax, y.max())
                x = np.arange(len(y)) + 1
                ax.plot(x, y,
                        'o-',
                        color=cmap[cond],
                        lw=2,
                        alpha=0.6,
                        label=cond)
            if 'combo' in gt:
                title = ct+' VJ combo'
            elif 'clonotype' in gt:
                title = ct+' clonotype'
            else:
                title = ct+' '+gt[0].upper()+' gene'
            ax.set_title(title)
            ax.grid(True)
            if ir == 1:
                ax.set_xlabel('rank')
            if ic == 0:
                ax.set_ylabel('clone size')
            ax.set_xlim(left=0.9)
            ax.set_ylim(0.9, 1.1 * ymax)
            ax.set_yscale('log')
    axs[0, -1].legend(
            loc='upper left',
            bbox_to_anchor=(1, 1),
            bbox_transform=axs[0, -1].transAxes,
            )
    fig.tight_layout()
    #fig.savefig('../../figures/BCR_rank_plots_by_condition.png')

    plt.ion()
    plt.show()
