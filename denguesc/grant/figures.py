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

sys.path.append(os.path.abspath('../../'))
from denguesc.ingest.ingest_for_antibodies import get_antibody_sequences


fig_fdn = '../../../../grants/AU_Ideas_2021/figures'


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

    # Use the newer cell_meta by Zhiyuan
    cellnames = adata.obs_names
    fdn = '../../data/datasets/20201002_merged'
    obs_new = pd.read_csv(
            f'{fdn}/mergedata_20210304_obs.tsv', sep='\t', index_col=0,
            )
    # Align cells
    obs_new = obs_new.loc[cellnames]
    for col in obs_new.columns:
        adata.obs[col] = obs_new[col]

    if False:
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

        df = adata.obs[['platform', 'cell_type']].groupby(['platform', 'cell_type']).size().unstack(0).rename(columns={'plate': 'PNAS 2018', '10X': 'preliminary 2021'})
        cmap = {'PNAS 2018': 'grey', 'preliminary 2021': 'tomato'}
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        for i, col in enumerate(['PNAS 2018', 'preliminary 2021']):
            y = np.arange(len(cts)) - 0.25 + 0.5 * i
            x = df.loc[cts, col]
            ax.barh(y, x, color=cmap[col], height=0.4, label=col)
        #ax.set_xscale('log')
        ax.set_yticks(np.arange(len(cts)))
        ax.set_yticklabels(cts)
        ax.set_ylim(len(cts) - 0.5, -0.5)
        ax.legend(loc='lower right')
        ax.set_xlabel('N. of sequenced cells')
        ax.set_xticks([0, 10000, 20000])
        ax.set_xticklabels(['0', '10k', '20k'])
        fig.tight_layout()
        fig.savefig(f'{fig_fdn}/n_cells_old_new.svg')

    if False:
        seq_meta = get_antibody_sequences()
        df = seq_meta['patient'].value_counts()
        fig, ax = plt.subplots(figsize=(3, 2))
        y = np.arange(len(df))
        x = df.values[::-1]
        ax.barh(y, x, height=0.8)
        ax.set_yticks([])
        ax.set_xlabel('N. of antibody sequences')
        ax.set_ylabel('Patients')
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(f'{fig_fdn}/n_antibody_sequences.svg')

    if False:
        from scipy.stats import gaussian_kde
        genes = ['CD163', 'CXCL10']
        conds = ['Healthy', 'dengue', 'S_dengue']
        amono = adata[adata.obs['cell_type'] == 'Monocytes']
        amcond = {k: amono[amono.obs['Condition'] == k] for k in conds}
        fig, axs = plt.subplots(3, 2, figsize=(3, 3), sharex=True)
        cmap = {'Healthy': 'steelblue', 'dengue': 'orange', 'S_dengue': 'darkred'}
        labeld = {'Healthy': 'Healthy', 'dengue': 'Dengue', 'S_dengue': 'SD'}
        for gene, axc in zip(genes, axs.T):
            axc[0].set_title(gene)
            for ax, cond in zip(axc, conds):
                xpoint = np.asarray(amcond[cond][:, [gene]].X.toarray())[:, 0]
                xpoint = 1e4 * xpoint / amcond[cond].obs['n_counts']
                xpoint = np.log10(xpoint + 0.1)
                x = np.linspace(-1, 4)
                kernel = gaussian_kde(xpoint, bw_method=0.5)
                y = kernel(x)
                ax.fill_between(x, 0, y, color=cmap[cond], label=labeld[cond], alpha=0.6)
                if gene == genes[-1]:
                    ax.legend()
                ax.set_yticks([])
            ax.set_xticks([-1, 0, 1, 2, 3])
            ax.set_xticklabels(['$0$', '$1$', '$10$', '$10^2$', '$10^3$'])
        axs[-1, 0].arrow(0.1, 0.6, 0.15, 0, head_width=0.12, head_length=0.12, color='k', overhang=0.2, transform=axs[-1, 0].transAxes)
        axs[-1, 1].arrow(0.4, 0.3, -0.12, 0, head_width=0.12, head_length=0.12, color='k', overhang=0.2, transform=axs[-1, 1].transAxes)
        fig.text(0.04, 0.53, 'Density of monocytes',
                 rotation=90, ha='left', va='center')
        fig.text(0.52, 0.02, 'Gene expression [cptt]', ha='center')
        fig.tight_layout(h_pad=0, w_pad=0, rect=(0.05, 0.05, 1, 1))
        fig.savefig(f'{fig_fdn}/monocytes_CD163_CXCL10.svg')

    if True:
        print('Look at virus infected cells')
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
        fn_viral_reads = '../../data/datasets/20200809_20kids/viral_reads/DWS_virus_obs.tsv'
        viral_reads = pd.read_csv(fn_viral_reads, sep='\t', index_col=0)
        nvr = viral_reads.groupby('cell_type').sum().loc[cts]
        nvr['+/- ratio'] = nvr['DENV_plus'] / nvr['DENV_reads']
        nvr['+/- ratio_comp'] = 1 - nvr['+/- ratio']
        sta = nvr.groupby('cell_type').sum()
        fig, axs = plt.subplots(1, 2, figsize=(3.7, 2), sharey=True, gridspec_kw={'width_ratios': [2, 1]})
        y = np.arange(len(cts))
        ax = axs[0]
        ax.barh(y, nvr['DENV_reads'], color='grey')
        ax.set_yticks(y)
        ax.set_yticklabels(cts)
        ax.set_xlabel('N. of DENV reads')
        ax.set_xscale('log')
        ax = axs[1]
        ax.barh(y, nvr['+/- ratio'], left=0, color='tomato')
        ax.barh(y, nvr['+/- ratio_comp'], left=nvr['+/- ratio'], color='steelblue')
        for i, ct in enumerate(cts):
            ax.plot([nvr.at[ct, '+/- ratio']] * 2, [i - 0.45, i + 0.45], color='k')
        ax.set_ylim(len(y) - 0.5, -0.5)
        ax.set_xlabel('+/- strand [%]')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['0', '50', '100'])
        fig.tight_layout(w_pad=0)
        fig.savefig(f'{fig_fdn}/viral_reads.svg')
