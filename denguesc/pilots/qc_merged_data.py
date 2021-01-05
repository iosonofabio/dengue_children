# vim: fdm=indent
'''
author:     Fabio Zanini
date:       02/10/20
content:    QC merged adult and kids data
'''
import os
import sys
import numpy as np
import pandas as pd

import anndata
import scanpy as sp

import matplotlib.pyplot as plt
import seaborn as sns



def split(adata, column):
    '''Split an AnnData by a column'''
    res = {}
    cats = adata.obs[column].cat.categories
    for cat in cats:
        res[cat] = adata[adata.obs[column] == cat]
    return res


if __name__ == '__main__':

    if False:
        print('Read loom file with adults and kids')
        fn = '../../data/datasets/20201002_merged/mergedata_20200930.loom'
        adata = anndata.read_loom(fn)

        print('High/low quality cells')
        df = adata.obs.copy()
        df['c'] = 1
        dfq = df[['cell_quality', 'ID', 'c']].groupby(['ID', 'cell_quality']).count()['c'].unstack(fill_value=0)
        dfq['frac_high'] = 1.0 * dfq['high'] / (dfq['high'] + dfq['low'])
        dfq.sort_values('frac_high', inplace=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(dfq.shape[0])
        ax.plot(
                x,
                100 * dfq['frac_high'],
                lw=2, marker='o',
                color='darkgrey',
                )
        ax.set_xticks(x)
        ax.set_xticklabels(dfq.index, rotation=90, fontsize=10)
        ax.grid(axis='y')
        ax.set_ylabel('High-quality cells [%]')
        fig.tight_layout()

        print('Select only high-quality cells and save')
        fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
        idx = adata.obs['cell_quality'] == 'high'
        adatah = adata[idx]
        adatah.write(fn_h5ad)

    if False:
        print('Load high-quality cells only')
        fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
        adatah = anndata.read_h5ad(fn_h5ad)

        print('Normalize')
        sp.pp.normalize_total(adatah, target_sum=1e6)

        print('Check rough cell type abundance')
        print(adatah.obs['cell_type'].value_counts())

        print('Look at those "platelets"')
        sys.path.append('/home/fabio/university/PI/projects/anndata_kolmogorov_smirnov/build/lib')
        import anndataks
        anndataks.rc['use_experimental_ks_2samp'] = True

        import pickle
        fn_comps = '../../data/datasets/20201002_merged/degs_platelets.pkl'
        if not os.path.isfile(fn_comps):
            adatap = split(adatah, 'cell_type')
            adata_pl = adatap['Platelets']
            comps = {}
            for cat, adatai in adatap.items():
                if cat == 'Platelets':
                    continue

                print(f'Comparing platelets with {cat}')
                adata1 = adatap['Platelets']
                adata1 = adata1[np.random.randint(adata1.shape[0], size=50)]
                adata2 = adatai[np.random.randint(adatai.shape[0], size=50)]
                comp = anndataks.compare(adata2, adata1)
                comp.rename(columns={
                    'avg1': f'avg_{cat}',
                    'avg2': 'avg_platelets',
                }, inplace=True)

                comps[cat] = comp

            with open(fn_comps, 'wb') as f:
                pickle.dump(comps, f)
        else:
            with open(fn_comps, 'rb') as f:
                comps = pickle.load(f)

        degs = {}
        for ct, comp in comps.items():
            fns = comp.index
            pfxs = ['TRAV', 'TRAJ', 'TRBV', 'TRBJ',
                    'IGHV', 'IGHJ', 'IGHD', 'IGLV', 'IGLJ', 'IGKV', 'IGKJ',
                    'MT-', 'RPS', 'RPL', 'MALAT1', 'HLA-A', 'HLA-B', 'HLA-C']
            idx = fns != ''
            for pfx in pfxs:
                idx &= ~fns.str.startswith(pfx)

            comp = comp.loc[idx]
            deg_up = comp.nlargest(20, 'statistic')
            deg_down = comp.nsmallest(20, 'statistic').iloc[::-1]
            deg_both = pd.concat([deg_up, deg_down])
            degs[ct] = deg_both

        with pd.ExcelWriter('../../data/gene_lists/deg_platlets.xlsx') as f:
            for ct, deg in degs.items():
                deg.to_excel(f, sheet_name=ct)

        if False:
            print('Check a bunch of genes in platelets for complementarity with the TSO')
            sys.path.append('/home/fabio/university/phd/sequencing/libraries/seqanpy/build/lib.linux-x86_64-3.8')
            import seqanpy
            import Bio.SeqIO
            from Bio.Seq import reverse_complement as rc
            seq_tso = 'AAGCAGTGGTATCAACGCAGAGTACATGGG'
            seq_oligodt = 'AAGCAGTGGTATCAACGCAGAGTAC' # exclude the polyT, otherwise it's obvious
            seq_oligo = seq_tso
            gene_seqs = {'up': {}, 'control': {}}
            fdn = '../../data/gene_sequences/platelets/'
            controls = ['VIM', 'SEC11C', 'PFN1', 'FAU', 'FCN1', 'TPT1', 'CTSS', 'RACK1']
            for fn in os.listdir(fdn):
                gene = fn.split('.')[0]
                seq = str(Bio.SeqIO.read(f'{fdn}{fn}', 'fasta').seq)
                if gene in controls:
                    sub = 'control'
                else:
                    sub = 'up'
                gene_seqs[sub][gene] = seq

            scores = {'up': [], 'control': []}
            for sub, gene_seq_sub in gene_seqs.items():
                print('-'*50)
                print(sub.upper())
                print('-'*50)
                for gene, seq in gene_seq_sub.items():
                    score, ali_gene, ali_tso = seqanpy.align_local(seq, seq_oligo)
                    score_rc, ali_gene_rc, ali_tso_rc = seqanpy.align_local(
                            seq, rc(seq_oligo),
                            score_mismatch=-5,
                            )
                    ali_gene_rc = rc(ali_gene_rc)
                    ali_tso_rc = rc(ali_tso_rc)
                    print(gene, score, score_rc)
                    print(f'{ali_gene}\n{ali_tso}\n\n{ali_gene_rc}\n{ali_tso_rc}\n')

                    scores[sub].append(max(score, score_rc))

            fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharex=True)
            cmap = {'up': 'tomato', 'control': 'steelblue'}
            for sub, sc in scores.items():
                x = np.sort(sc)
                y = 1.0 - np.linspace(0, 1, len(x))
                axs[0].plot(x, y, lw=2, label=sub, color=cmap[sub])
                xm = x.mean()
                xs = x.std()
                xgauss = np.linspace(30, 48, 1000)
                ygauss = np.exp(-((xgauss - xm)**2)/(xs**2))
                axs[1].plot(xgauss, ygauss, lw=2, label=sub, color=cmap[sub])
            axs[0].set_xlabel('alignment score')
            axs[1].set_xlabel('alignment score')
            axs[0].grid(True)
            axs[1].grid(True)
            axs[0].legend()
            fig.tight_layout()

            print('Alright, genes up in "platelets" are probably binding to the TSO')

    if False:
        print('Keep only decent cell types')
        fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
        adatah = anndata.read_h5ad(fn_h5ad)

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
        idx = adatah.obs['cell_type'].isin(cts)
        adatag = adatah[idx]
        adatag.write(fn_h5ad)

    if False:
        print('Check cell type abundance')
        fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
        adata = anndata.read_h5ad(fn_h5ad)

        df = adata.obs[['Age', 'Gender', 'Condition', 'ID', 'cell_type']].copy()

        print('Make a patient table with cell counts')
        ptable = df.groupby(df.columns.tolist()[:-1]).size()
        ptable = ptable[ptable > 0]
        cell_counts = df.groupby(df.columns.tolist()).size().unstack().loc[ptable.index]

        print('Number of patients with 5+ cells per type, divided by condition')
        n_pats = (cell_counts >= 5).unstack('Condition', fill_value=False).sum(axis=0).unstack().astype(int)

    if True:
        print('Quick check on biomarkers')
        fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
        adata = anndata.read_h5ad(fn_h5ad)
        sp.pp.normalize_total(adatah, target_sum=1e6)
        obs = adata.obs

        biomarkers = [
            ('CD163', 'Monocytes'),
            ('MX2', 'B_cells'),
            ('MS4A1', 'B_cells'),
            ('CD14', 'Monocytes'),
            ('CD3E', 'T_cells'),
        ]
        conditions = ['Healthy', 'dengue', 'DWS', 'S_dengue']
        plats = ['plate', '10X']
        exps = {}
        for (gene, ct) in biomarkers:
            for plat in plats:
                for cond in conditions:
                    idx = (obs['cell_type'] == ct) & (obs['Condition'] == cond) & (obs['platform'] == plat)
                    x = adata[idx, gene].X.toarray()[:, 0]
                    x.sort()
                    exps[(gene, ct, plat, cond)] = x

        fig_fdn = '../../figures/shared/quick_predictor_check/'
        cmap = {'Healthy': 'mediumblue', 'dengue': 'darkolivegreen', 'DWS': 'darkorange', 'S_dengue': 'tomato'}
        for (gene, ct) in biomarkers:
            fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
            for ax, plat in zip(axs, plats):
                for cond in conditions:
                    x = exps[(gene, ct, plat, cond)] + 0.1
                    y = 1.0 - np.linspace(0, 1, len(x))
                    ax.plot(x, y, lw=2, color=cmap[cond], label=cond)
                ax.set_xlabel('Gene expression [cpm]')
                ax.grid(True)
                ax.set_xlim(left=0.09)
                ax.set_xscale('log')
                #ax.set_yscale('log')
                ax.set_title(plat)
                if ax == axs[-1]:
                    ax.legend(
                            loc='upper left',
                            bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,
                            )
                else:
                    ax.set_ylabel('Fraction of cells\nexpressing > x')
            fig.suptitle(f'{gene} in {ct}')
            fig.tight_layout()
            fxf = f'{fig_fdn}{gene}_{ct}.png'
            fig.savefig(fxf)
