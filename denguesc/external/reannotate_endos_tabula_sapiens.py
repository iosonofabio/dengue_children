# vim: fdm=indent
'''
author:     Fabio Zanini
date:       09/06/21
content:    Reannotate endos tabula sapiens (sigh)
'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import anndata

sys.path.append('/home/fabio/university/postdoc/singlet')
import singlet


if __name__ == '__main__':

    fn_endo = '../../data/tabula_sapiens/endothelial/TS_Endothelial.h5ad'
    #adata = anndata.read_h5ad(fn_endo)
    ds = singlet.Dataset(
        dataset={
            'path': fn_endo,
        },
    )
    ds.obs['coverage'] = ds.counts.sum(axis=0)
    ds.counts.normalize('counts_per_ten_thousand', inplace=True)

    print('Annotate organs')
    organs = ds.obs['Organ'].value_counts().index

    dss2 = ds.query_samples_by_metadata('Method == "smartseq2"')
    ds10x = ds.query_samples_by_metadata('Method != "smartseq2"')

    print('Check current annotations')
    cell_types = ds.obs['Annotation'].value_counts().index
    genes = ['PECAM1', 'CDH5', 'GJA5', 'BMX', 'VWF', 'CA4', 'CA8', 'MKI67', 'THY1', 'CCL21']
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(genes), 1 + 0.8 * len(cell_types)))
    ds.plot.dot_plot(
        group_by='Annotation',
        plot_list=genes,
        layout='horizontal',
        threshold=0.1,
        cmap='plasma',
        ax=ax,
    )
    fig.tight_layout()

    plt.ion(); plt.show()

    print('Exclude Car4+ caps and lymphatics')
    cts_include = [
            'capillary endothelial cell',
            'endothelial cell of vascular tree',
            'endothelial cell',
            'vein endothelial cell',
            #'capillary aerocyte',
            'lung microvascular endothelial cell',
            'endothelial cell of artery',
            #'endothelial cell of lymphatic vessel',
            'gut endothelial cell',
    ]
    dsm = ds.query_samples_by_metadata(
        'Annotation in @cts_include', local_dict=locals(),
    )

    print('Exclude useless features')
    prefixes_exclude = [
        'AL', 'AC', 'AP',
        'RNU', 'RPL', 'RPS',
        'MT-',
        'LINC', 'MIR',
        'IGLV', 'IGHV', 'IGHJ', 'IGLJ', 'IGKV', 'IGKJ',
    ]
    def exclude_all(features, prefixes):
        fu = features.str.startswith
        idx = np.zeros(len(features), bool)
        for pref in prefixes:
            idx |= fu(pref)
        idx = ~idx
        return features[idx]
    features = exclude_all(dsm.featurenames, prefixes_exclude)
    dsf = dsm.query_features_by_name(features)

    print('Unbiased feature selection')
    feature_sel = dsf.feature_selection.overdispersed()

    print('Biased feature selection')
    fea_corr = {}
    ncorr = 100
    gene_seeds = ['GJA5', 'BMX', 'CCL21', 'CA8', 'CA4', 'SOD2']
    corr = dss2.correlation.correlate_features_features(
            features='all',
            features2=gene_seeds,
            fillna=0,
            ).fillna(0)
    for col in gene_seeds:
        tmp = corr[col].nlargest(ncorr // 2)
        tmp = tmp.loc[tmp > 0.1]
        tmp2 = corr[col].nsmallest(ncorr // 2)
        tmp2 = tmp2.loc[tmp2 < -0.1]
        fea_corr[col] = tmp.index.tolist() + tmp2.index.tolist()
    feature_sel = list(set(sum(fea_corr.values(), [])))

    print('Feature selection')
    dsf = dsm.query_features_by_name(feature_sel)

    print('PCA')
    dspca = dsf.dimensionality.pca(n_dims=40, return_dataset='samples')

    print('UMAP')
    vs = dspca.dimensionality.umap()

    dsm.obs['Annotation_Fabio'] = ''

    if False:
        print('Poor man\'s clustering')
        idx1 = vs.index[(vs['dimension 1'] < -6)]# & (vs['dimension 2'] > 3)]
        idx2 = vs.index[vs['dimension 1'] > -6]
        ds1 = dsm.query_samples_by_name(idx1)
        ds2 = dsm.query_samples_by_name(idx2)
        ds1s = ds1.subsample(100)
        ds2s = ds2.subsample(100)
        comp = ds1s.compare(ds2s, method='kolmogorov-smirnov-rich')

        dsm.obs.loc[idx1, 'Annotation_Fabio'] = 'type1'
        dsm.obs.loc[idx2, 'Annotation_Fabio'] = 'type2'
        dsi = dsm.query_samples_by_metadata('Annotation_Fabio in ("type1", "type2")')

        compf = comp.loc[features]
        genes = compf.nlargest(10, 'statistic').index.tolist()
        nup = len(genes)
        genes += compf.nsmallest(10, 'statistic').index.tolist()

        fig, ax = plt.subplots(figsize=(2, 1 + 0.4 * len(genes)))
        dsi.plot.dot_plot(
            group_by='Annotation_Fabio',
            plot_list=genes,
            layout='vertical',
            threshold=0.1,
            cmap='plasma',
            ax=ax,
        )
        ax.axhline(nup - 0.5, lw=1.5, color='grey')
        fig.tight_layout()

    print('Plot umap')
    genes = [
        #'PECAM1', 'CDH5',
        'GJA5',
        #'BMX',
        'VWF',
        #'MKI67', 'CA4', 'CCL21',
        #'ACTB',
        #'CHP1',
        'IFITM1',
        #'SBNO2',
        #'DUSP3',
        #'FGF2',
        #'CD74',
        'TGFBR2',
        'ADIRF',
        'SHANK3',
        #'TPT1',
        ]
    genes = sum([x[:3] + x[-3:] for x in fea_corr.values()], [])

    genes += [
        'Annotation', 'Donor', 'Organ',
        #'Method',
        #'Annotation_Fabio',
        ]

    nrows = 3

    cmaps = {
        'Annotation': dict(zip(cell_types, sns.color_palette('husl', n_colors=len(cell_types)))),
        'Organ': dict(zip(organs, sns.color_palette('husl', n_colors=len(organs)))),
    }
    ncols = len(genes) // nrows + int(bool(len(genes) % nrows))
    fig, axs = plt.subplots(
            nrows, ncols,
            figsize=(0.8 + 1 * ncols, 0.8 + 1 * nrows),
            )
    axs = axs.ravel()
    for ax, gene in zip(axs, genes):
        dsm.plot.scatter_reduced(
            vs,
            color_by=gene,
            color_log=None,
            ax=ax,
            alpha=0.01 + 0.15 * (dsm.n_samples < 3000),
            s=10,
            cmap=cmaps.get(gene, 'viridis'),
            #high_on_top=True,
        )
        ax.set_title(gene)
        ax.set_axis_off()
        if gene in ('Annotation', 'Organ'):
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            for ct in cmaps[gene]:
                ax2.scatter([], [], label=ct, color=cmaps[gene][ct])
            ax2.legend()
            ax2.set_axis_off()
            fig2.tight_layout()

    fig.tight_layout()
