# vim: fdm=indent
'''
author:     Fabio Zanini
date:       02/10/20
content:    Compare gene expression changes in kids and adults
'''
import os
import sys
import numpy as np
import pandas as pd

import anndata
import scanpy as sp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def split(adata, column):
    '''Split an AnnData by a column'''
    res = {}
    cats = adata.obs[column].cat.categories
    for cat in cats:
        res[cat] = adata[adata.obs[column] == cat]
    return res


def average(adata, columns):
    from itertools import product
    res = {}
    if isinstance(columns, str):
        columns = [columns]
    cats_list = [adata.obs[c].cat.categories for c in columns]
    for comb in product(*cats_list):
        idx = pd.Series(np.ones(adata.shape[0], bool), index=adata.obs_names)
        for i in range(len(columns)):
            idx &= adata.obs[columns[i]] == comb[i]
        adatai = adata[idx]
        if adatai.shape[0] < 10:
            continue
        avi = np.asarray(adatai.X.mean(axis=0))[0]
        res[comb] = pd.Series(avi, index=adata.var_names)
    return pd.DataFrame(res)


if __name__ == '__main__':

    print('Load and keep only decent cell types')
    fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
    adata = anndata.read_h5ad(fn_h5ad)

    cts = [
        'B_cells',
        'Monocytes',
        'NK_cells',
        'Plasmablasts',
        #'Platelets',  # TSO binders
        'T_cells',
        'cDCs',
        #'doublets',  # doublets
        'pDCs',
        #'unhealthy_cells',  # immediate-early genes
        ]
    idx = adata.obs['cell_type'].isin(cts)
    adata = adata[idx]

    adata.obs['Infected'] = pd.Categorical(
            (adata.obs['Condition'] != 'Healthy').map({True: 'Infected', False: 'Healthy'}),
            categories=['Infected', 'Healthy'])
    adata.obs['Age_class'] = pd.Categorical(
            (adata.obs['Age'] > 18).map({True: 'Adult', False: 'Child'}),
            categories=['Adult', 'Child'])

    cats = [
        ('Infected', 'Infected', 'Healthy', 'Infected (all) versus healthy'),
        ('Condition', 'S_dengue', 'dengue', 'Severe versus dengue'),
        ]

    for cat, catpos, catneg, title in cats:
        res = average(adata, [cat, 'cell_type', 'Age_class'])
        log2_fc = np.log2(res[catpos] + 0.1) - np.log2(res[catneg] + 0.1)

        # Some genes got renamed by Zhiyuan...
        log2_fc.loc['IFI27'] = 0.5 * (log2_fc.loc['IFI27'] + log2_fc.loc['IFI27_1'])
        log2_fc.drop('IFI27_1', axis=0, inplace=True)

        flt_prefix = [
                'IGHV', 'IGHD', 'IGHJ', 'IGKV', 'IGKJ', 'IGLV', 'IGLJ',
                'TRAV', 'TRAJ', 'TRBV', 'TRBJ',
                'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 'HLA-DRB', 'HLA-DQ', 'HLA-E',
                'MT-',
                ]
        idx = pd.Series(np.ones(len(log2_fc), bool), index=log2_fc.index)
        for pref in flt_prefix:
            idx &= ~idx.index.str.startswith(pref)
        log2_fcp = log2_fc.loc[idx]

        log2_fcp.to_csv(
            f'../../data/gene_lists/log2_fold_change_kids_adults_{catpos}_VS_{catneg}.tsv.gz',
            sep='\t', compression='gzip')
        log2_fcp.to_excel(
            f'../../data/gene_lists/log2_fold_change_kids_adults_{catpos}_VS_{catneg}.xlsx',
            )
        continue

        fig, axs = plt.subplots(1, 5, figsize=(17, 5), sharex=True, sharey=True)
        axs = axs.ravel()
        for ct, ax in zip(['B_cells', 'Monocytes', 'NK_cells', 'T_cells', 'Plasmablasts'], axs):
            x = log2_fcp[(ct, 'Child')]
            y = log2_fcp[(ct, 'Adult')]
            r = np.sqrt(x**2 + y**2)
            r /= r.max()
            c = np.ones((len(r), 4))
            c[:] = matplotlib.colors.to_rgba('steelblue')
            c[:, -1] = 0.01 + 0.99 * r**2
            s = 5 + 95 * r**2
            ax.scatter(x, y, s=s, c=c, zorder=10)
            topx = list(x.nlargest(5).index)
            topy = list(y.nlargest(5).index)
            botx = list(x.nsmallest(5).index)
            boty = list(y.nsmallest(5).index)
            topd = list((y - x).nlargest(5).index)
            botd = list((y - x).nsmallest(5).index)
            for gene in topx + topy + botx + boty + topd + botd:
                ax.scatter(x[gene], y[gene], s=30, facecolor='none', edgecolor='k', zorder=11)
                ax.text(x[gene], y[gene], gene, ha='center', va='bottom')

            ax.set_title(ct)
            ax.grid(True)
            ax.set_xlabel('log2FC (Children)')
            ax.set_ylabel('log2FC (Adults)')
        fig.suptitle(title)
        fig.tight_layout()
        #fig.savefig(f'../../figures/example_scatter_genes {title}.png')
