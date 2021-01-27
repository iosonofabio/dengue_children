# vim: fdm=indent
'''
author:     Fabio Zanini
date:       05/01/21
content:    Optimize code that looks for interactions.
'''
import os
import sys
import numpy as np
import pandas as pd

import anndata
import scanpy as sp

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/fabio/university/PI/projects/anndata_utils/')
from anndata_utils.partition import split, expressing_fractions, average


pdict = {'child': [
    '1_019_01',
    '3_012_01',
    '3_037_01',
    '6_023_01',
    '3_047_01',
    '3_074_01',
    '5_030_01',
    '5_193_01',
    '5_154_01',
    '6_001_01',
    '5_041_01',
    '1_075_01',
    '1_140_01',
    '1_144_01',
    '5_044_01',
    '1_002_01',
    '6_020_01',
    '6_025_01',
    '6_028_01',
    '5_089_01'],
 'adult': [
     '3_013_01',
     '3_027_01',
     '1_008_01',
     '1_013_01',
     '1_020_01',
     '1_026_01',
     '3_018_01',
     '3_006_01',
     '1_010_01',
     '1_036_01'],
 }


def get_interactions(
            fracd, avgd,
            criterion,
            redundant=True,
            ):
    from collections import defaultdict

    th = criterion['threshold']
    cell_types = list(obs['cell_type'].cat.categories)
    res = []
    for col in fracd.columns:
        datas, cond, cell_type1 = col
        for cell_type2 in cell_types:
            col2 = (datas, cond, cell_type2)
            fra = fracd.loc[ga, col].values
            frb = fracd.loc[gb, col2].values
            avga = avgd.loc[ga, col].values
            avgb = avgd.loc[gb, col2].values
            key = criterion['key']
            if isinstance(th, dict):
                th1 = th[col[0]]
            else:
                th1 = th

            ind = (stats[key].loc[ga, col].values > th1) & (stats[key].loc[gb, col2].values > th1)
            ind = ind.nonzero()[0]
            for i in ind:
                resi = {
                    'dataset': datas,
                    'Condition': cond,
                    'cell_type1': cell_type1,
                    'cell_type2': cell_type2,
                    'gene_name_a': interactions.iloc[i]['gene_name_a'],
                    'gene_name_b': interactions.iloc[i]['gene_name_b'],
                    'frac1': fra[i],
                    'frac2': frb[i],
                    'avg1': avga[i],
                    'avg2': avgb[i],
                }
                res.append(resi)
    res = pd.DataFrame(res)

    res['frac_sum'] = res['frac1'] + res['frac2']

    if not redundant:
        return res

    # Make it redundant (Yike calls it 'merge')
    res2 = res.loc[res['cell_type1'] != res['cell_type2']].copy()
    res2.rename(columns={
        'cell_type1': 'cell_type2',
        'cell_type2': 'cell_type1',
        'gene_name_a': 'gene_name_b',
        'gene_name_b': 'gene_name_a',
        'frac1': 'frac2',
        'frac2': 'frac1',
        'avg1': 'avg2',
        'avg2': 'avg1',
    }, inplace=True)
    res2 = res2[res.columns]

    resr = pd.concat([res, res2], axis=0)

    return resr


def loc_gene_ct(fracd, gene, cell_type):
    return fracd.loc[gene, pd.IndexSlice[:, :, cell_type]]


if __name__ == '__main__':

    print('Load interactions')
    fn_int = '../../data/interactions/interaction_unpacked_mouse.tsv'
    interactions = pd.read_csv(fn_int, sep='\t')[['gene_name_a', 'gene_name_b']]
    ga, gb = interactions['gene_name_a'], interactions['gene_name_b']

    if True:
        print('Load high-quality cells only')
        fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
        adata = anndata.read_h5ad(fn_h5ad)
        adata.obs['dataset'] = adata.obs['platform'].replace({
            '10X': 'child',
            'plate': 'adult',
        })
        adata.obs.rename(columns={'ID': 'Patient'}, inplace=True)
        pdata = (adata.obs[['Patient', 'dataset', 'Condition']]
                      .drop_duplicates()
                      .set_index('Patient'))
        pdata['is_sick'] = pdata['Condition'] != 'Healthy'

        sp.pp.normalize_total(adata, target_sum=1e6)

    print('Restrict to interaction genes')
    genes = np.unique(interactions)
    adatag = adata[:, genes]

    print('Split by cell type, adult and children, and condition')
    obs = adatag.obs
    adatag.obs['split_col'] = obs['dataset'] + '+' + obs['Condition'].astype(str) + '+' + obs['cell_type'].astype(str)

    fracd = expressing_fractions(adatag, ['dataset', 'Condition', 'cell_type'])
    avgd = average(adata, ['dataset', 'Condition', 'cell_type'], log=False)
    stats = {
        'frac_exp': fracd,
        'avg_exp': avgd,
    }

    # Flexible criterion
    criterion = {'key': 'frac_exp', 'threshold': 0.1}
    criterion = {'key': 'avg_exp', 'threshold': {'child': 60, 'adult': 35}}
    criterion = {'key': 'avg_exp', 'threshold': 50}

    int_data = get_interactions(fracd, avgd, criterion)

    avgdp = average(adata, ['Patient', 'cell_type'], log=False)

    print('Find robust upregulation in pairwise patient comparisons')
    def get_pairwise_comparisons(avgdp, pdata, nrep=100):
        cts = ['B_cells', 'T_cells', 'Monocytes', 'NK_cells']
        gediff = np.zeros((len(avgdp), len(cts) * nrep))
        cols = [[], []]
        gby = {key: val for key, val in pdata.groupby('is_sick')}
        j = 0
        for i in range(nrep):
            # Healthy patient
            ph = gby[False]
            ph = ph.index[np.random.randint(len(ph))]

            # Sick patient
            ps = gby[True]
            ps = ps.index[np.random.randint(len(ps))]

            # Not all patients have all cell types
            tmps = avgdp[ps]
            tmph = avgdp[ph]
            ctsi = np.intersect1d(tmps.columns, tmph.columns)
            ctsi = np.intersect1d(ctsi, cts)

            # Diff exp (log2 fc)
            tmp = np.log2(tmps[ctsi] + 0.1) - np.log2(tmph[ctsi] + 0.1).values
            gediff[:, j: j + len(ctsi)] = tmp.values
            cols[1].extend([ps+' VS '+ph] * len(ctsi))
            cols[0].extend(ctsi)
            j += len(ctsi)

        gediff = gediff[:, :j]
        gediff = pd.DataFrame(
            gediff, index=avgdp.index,
            columns=pd.MultiIndex.from_arrays(cols),
            )
        return gediff

    # Example calls
    gediff = get_pairwise_comparisons(avgdp, pdata)
    (gediff.loc[genes, 'NK_cells'] > 0).sum(axis=1).nlargest(30)
    get_pairwise_comparisons(avgdp, pdata.query('dataset == "child"'))


    print('Plot pairwise comparisons')
    def plot_pairwise_comparison_top_genes(gediff, ngenes=10):
        from scipy.stats import gaussian_kde

        cts = ['B_cells', 'T_cells', 'Monocytes', 'NK_cells']
        colors = sns.color_palette('husl', n_colors=ngenes)
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 5))
        axs = axs.ravel()
        for ct, ax in zip(cts, axs):
            genesp = (gediff.loc[genes, ct] > 0).sum(axis=1).nlargest(ngenes).index
            for i, gene in enumerate(genesp):
                x = gediff.loc[gene, ct].clip(-10, 10).values
                xfit = np.linspace(-10, 10, 100)
                y = gaussian_kde(x, bw_method=0.3)(xfit)
                y /= 1.1 * y.max()
                yb = i + 1
                yp = yb - y

                ind = xfit >= 0
                ind2 = xfit <= xfit[ind].min()
                ax.plot(xfit, yp, color=colors[i], alpha=0.8, lw=1)
                ax.fill_between(xfit[ind], yb, yp[ind],
                        facecolor=colors[i], edgecolor='none', alpha=0.7)
                ax.fill_between(xfit[ind2], yb, yp[ind2],
                        facecolor=colors[i], edgecolor='none', alpha=0.2)

                ax.text(-9.8, yb - 0.5, gene, va='center')
                ax.axhline(yb, lw=1, color='grey', alpha=0.5)
            ax.grid(True)
            ax.set_xlim(-10, 10)
            ax.set_ylim(len(genesp) + 0.1, -0.1)
            ax.set_yticks(np.arange(len(genesp)) + 1)
            ax.set_title(ct)
            if ct not in cts[:2]:
                ax.set_xlabel('Log2 [Fold change]\nsick VS ctrl')
            if ct in cts[::2]:
                ax.set_ylabel('Rank\n[% pairwise comparisons]')
        fig.tight_layout()
        return {'fig': fig, 'axs': axs}

    d = plot_pairwise_comparison_top_genes(gediff)

    gediffd = {}
    for key in ['child', 'adult']:
        gediffd[key] = get_pairwise_comparisons(avgdp, pdata.query('dataset == @key'))

    def plot_pairwise_comparison_top_genes_mirror(gediffd, ngenes=10, lead='adult'):
        from scipy.stats import gaussian_kde

        for key in gediffd:
            if key != lead:
                tail = key
                break
        else:
            raise KeyError('Only lead key in dict??')

        cts = ['B_cells', 'T_cells', 'Monocytes', 'NK_cells']
        colors = sns.color_palette('husl', n_colors=ngenes)
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 5))
        axs = axs.ravel()
        for ct, ax in zip(cts, axs):
            genesp = ((gediffd[lead].loc[genes, ct] > 0)
                                    .sum(axis=1)
                                    .nlargest(ngenes)
                                    .index)
            for i, gene in enumerate(genesp):
                xfit = np.linspace(-10, 10, 100)
                for key in [lead, tail]:
                    x = gediffd[key].loc[gene, ct].clip(-10, 10).values
                    if len(np.unique(x)) > 1:
                        y = gaussian_kde(x, bw_method=0.3)(xfit)
                    else:
                        y = np.zeros_like(x)
                        y[np.argmin(np.abs(x[0] - xfit))] = 1
                    y /= 2 * 1.1 * y.max()
                    yfloor = i + 1

                    if key == lead:
                        yp = yfloor - y
                    else:
                        yp = yfloor + y

                    ind = xfit >= 0
                    ind2 = xfit <= xfit[ind].min()
                    ax.plot(xfit, yp, color=colors[i], alpha=0.8, lw=1)
                    ax.fill_between(
                            xfit[ind], yfloor, yp[ind],
                            facecolor=colors[i], edgecolor='none', alpha=0.7)
                    ax.fill_between(
                            xfit[ind2], yfloor, yp[ind2],
                            facecolor=colors[i], edgecolor='none', alpha=0.2)

                    if (ax == axs[0]) and (i == 0):
                        ax.text(9.2, yfloor - 0.05, lead, ha='right', va='bottom')
                        ax.text(9.2, yfloor + 0.05, tail, ha='right', va='top')

                ax.text(
                    -9.2, yfloor, gene, va='center',
                    bbox={'facecolor': 'white', 'edgecolor': 'none'})
                ax.axhline(yfloor, lw=1, color='grey', alpha=0.5)
            ax.grid(True)
            ax.set_xlim(-10, 10)
            ax.set_ylim(len(genesp) + 0.6, 0.5 - 0.1)
            ax.set_yticks(np.arange(len(genesp)) + 1)
            ax.set_title(ct)
            if ct not in cts[:2]:
                ax.set_xlabel('Log2 [Fold change]\nsick VS ctrl')
            if ct in cts[::2]:
                ax.set_ylabel('Rank\n[% pairwise comparisons]')
        fig.tight_layout()
        return {'fig': fig, 'axs': axs}

    plot_pairwise_comparison_top_genes_mirror(gediffd)

    cts = ['B_cells', 'T_cells', 'Monocytes', 'NK_cells']
    frpos = pd.concat(
        {key: pd.DataFrame({ct: (gediffd[key][ct] > 0).mean(axis=1) for ct in cts}) for key in ['child', 'adult']},
        axis=1)
    frneg = pd.concat(
        {key: pd.DataFrame({ct: (gediffd[key][ct] < 0).mean(axis=1) for ct in cts}) for key in ['child', 'adult']},
        axis=1)

    def scatter_fraction_positives(frpos, frneg, n_annotate=0, **kwargs):
        fig, axs = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)
        axs = axs.ravel()
        for ct, ax in zip(cts, axs):
            tmp = frpos.loc[:, pd.IndexSlice[['adult', 'child'], ct]]
            tmp2 = frneg.loc[:, pd.IndexSlice[['adult', 'child'], ct]]
            idx = ((tmp + tmp2) > 0.8).any(axis=1)
            tmp = tmp.loc[idx]
            #tmp = tmp.loc[np.sqrt((tmp**2).sum(axis=1)) > 0.5]
            x, y = tmp.values.T
            ax.scatter(x, y, alpha=min(1, 100. / len(frpos)))
            ax.grid(True)
            ax.set_title(ct)
            if ct in cts[::2]:
                ax.set_ylabel('Fraction > 0 children')
            if ct in cts[2:]:
                ax.set_xlabel('Fraction > 0 adults')

            if n_annotate:
                genes_anno = (tmp**2).sum(axis=1).nlargest(n_annotate).index
                txts = []
                for gene in genes_anno:
                    xi, yi = tmp.loc[gene].values
                    h = ax.text(xi, yi, gene, ha='center', va='bottom')
                    txts.append(h)

                from adjustText import adjust_text
                adjust_text(txts, ax=ax, **kwargs)

        fig.tight_layout()
        return {'fig': fig, 'axs': axs}

    scatter_fraction_positives(frpos, frneg)
