# vim: fdm=indent
'''
author:     Fabio Zanini
date:       13/10/21
content:    A few figures for Oz Single Cell 2021
'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import anndata
sys.path.insert(0, '/home/fabio/university/PI/projects//anndata_utils')
import anndata_utils
import scanpy as sp


fig_fdn = '../../figures/ozsinglecell2021/'


if __name__ == '__main__':

    print('Load count data')
    fn_loom = '../../data/datasets/20211007_allkids/counts/mergedata_20211001.loom'
    adata = anndata.read_loom(fn_loom)
    # obs['n_counts'] already has the total counts
    for ir, row in enumerate(adata.X):
        adata.X[ir] = 1e4 * row / row.sum()

    print('Plot relative fractions of subtypes')
    groups = [
        {'ct': 'Monocytes', 'cst': 'classical monocytes'},
        {'ct': 'Plasmablasts', 'cst': 'cycling Plasmablasts'},
        {'ct': 'NK/T_cells', 'cst': 'XCL_high NK cells', 'tot': [
            'XCL_low NK cells', 'XCL_high NK cells',
        ], 'totlabel': 'NK cells'},
        {'ct': 'NK/T_cells', 'cst': 'CD8+ effector T cells', 'tot': [
            'CD4+ T cells', 'CD8+ effector T cells', 'CD8+ naive/memory T cells',
        ], 'totlabel': 'T cells'},
    ]
    data_all = {}
    for ig, group in enumerate(groups):
        ct = group['ct']
        cst = group['cst']
        totlabel = group.get('totlabel', ct)

        adatam = adata[adata.obs['cell_type_new'] == ct]
        df = adatam.obs.groupby(['Condition', 'ID', 'cell_subtype_new']).size().unstack(fill_value=0)

        data = {}
        for cats in [('dengue', 'DWS'), ('S_dengue', )]:
            idx = df.index.get_level_values('Condition').isin(cats)
            dfi = df.loc[idx]
            for (_, pname), row in dfi.iterrows():
                if 'tot' in group:
                    den = row[group['tot']].sum()
                else:
                    den = row.sum()
                if den == 0:
                    continue
                frac = 1.0 * row[cst] / den
                data[pname] = {
                    'frac': frac,
                    'condition': cats[0],
                    }
        data = pd.DataFrame(data).T
        data['frac'] = data['frac'].astype(float)
        data_all[cst] = data

        from scipy.stats import gaussian_kde
        fig, ax = plt.subplots(figsize=(3, 3))
        cmap = {'dengue': 'steelblue', 'S_dengue': 'darkred'}
        labeld = {'dengue': 'non progressing', 'S_dengue': 'progressing'}
        #sns.violinplot(x="condition", y="frac", data=data, ax=ax, bw_method=0.2)
        for ic, cond in enumerate(['S_dengue', 'dengue']):
            datum = data.loc[data['condition'] == cond]['frac'].values
            ax.plot(np.sort(datum), 1.0 - np.linspace(0, 1, len(datum)),
                    label=labeld[cond], color=cmap[cond])
        #    ymodel = np.linspace(0, 1, 100)
        #    xmodel = gaussian_kde(datum, bw_method=0.1)(ymodel)
        #    ax.fill_betweenx(ymodel, ic, ic + xmodel)
        if ig == 0:
            ax.legend(loc='lower left')
        ax.grid(True)
        ax.set_xlabel(cst+'\n[Fraction of '+totlabel+']')
        ax.set_ylabel('Fraction of patients > x')
        fig.tight_layout()
        fig.savefig(fig_fdn+'frac_'+totlabel.lower().replace('/', '_')+'.png')

    pnames = list(set.union(*[set(x.index.tolist()) for x in data_all.values()]))
    tmp = df.reset_index(0).loc[pnames, ['Condition']].copy()
    for cst, data in data_all.items():
        tmp[cst] = np.nan
        for pname, frac in data['frac'].items():
            tmp.at[pname, cst] = frac
    dataa = tmp.dropna().copy()
    dataa['Condition'] = dataa['Condition'].replace('DWS', 'dengue')

    print('PCA')
    import umap 
    model = umap.UMAP(n_neighbors=3)
    vs = model.fit_transform(dataa.iloc[:, 1:])

    from sklearn.manifold import TSNE
    model = TSNE(perplexity=6)
    vs = model.fit_transform(dataa.iloc[:, 1:])

    from sklearn.decomposition import PCA
    model = PCA(n_components=2)
    vs = model.fit_transform(dataa.iloc[:, 1:])

    datap = pd.DataFrame(-vs, columns=['PC1', 'PC2'], index=dataa.index)
    datap['Condition'] = dataa['Condition']

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.scatterplot(
            data=datap, x='PC1', y='PC2', hue='Condition', ax=ax,
            palette=cmap,
            legend=False,
            alpha=0.7,
            zorder=10,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_fdn+'frac_PCA.png')

    print('ML on everythin')
    from sklearn.svm import SVC
    X = dataa.iloc[:, 1:].values
    y = (dataa['Condition'] == 'S_dengue').astype(int).values
    model = SVC(kernel='rbf', gamma='auto')
    #model.fit(X, y)

    print('Decision tree')
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()

    print('Random forest')
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()

    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import LeavePOut
    cv_results = cross_validate(
            model, X, y,
            cv=LeavePOut(3),
            scoring=('accuracy', 'precision', 'f1'),
            )

    print('ML on NK/T')
    from sklearn.svm import SVC
    X = dataa.iloc[:, -2:].values
    y = (dataa['Condition'] == 'S_dengue').astype(int).values
    model = SVC(
            C=10.0,
            kernel='poly',
            gamma=1e1,
            probability=True,
            )
    model.fit(X, y)

    # Predict grid
    xmin, ymin = X.min(axis=0) * 0
    xmax, ymax = X.max(axis=0) * 1.1
    ng = 500
    xm, ym = np.meshgrid(np.linspace(xmin, xmax, ng), np.linspace(ymin, ymax, ng))
    probs = model.predict_proba(np.dstack([xm, ym]).reshape((ng*ng, 2)))[:, 1].reshape((ng, ng))
    preds = model.decision_function(np.dstack([xm, ym]).reshape((ng*ng, 2))).reshape((ng, ng))

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    sns.scatterplot(
            data=dataa, x=dataa.columns[-2], y=dataa.columns[-1], hue='Condition', ax=ax,
            palette=cmap,
            legend=False,
            alpha=0.7,
            zorder=10,
            )
    #ax.contourf(
    #        xm, ym, probs, levels=100, alpha=0.5,
    #        cmap='Greys_r',
    #        )
    ax.contour(
            xm, ym, preds, alpha=0.5,
            colors='k',
            levels=[0],
            linestyles=['--'],
            zorder=7,
            )
    fig.tight_layout()
    fig.savefig(fig_fdn+'frac_SVM_NK_and_CD8T.png')

    print('Gene expression in monocytes')
    adatam = adata[adata.obs['cell_type_new'] == 'Monocytes']
    sp.pp.normalize_total(adatam, target_sum=1e4)
    adata_av = anndata_utils.partition.average(adatam, 'ID')
    pdict = adatam.obs[['ID', 'Condition']].copy().drop_duplicates().set_index('ID')['Condition']
    pdict = pdict.loc[pdict != 'Healthy'].replace('DWS', 'dengue')
    pdict_back = {key: pdict.index[pdict == key].tolist() for key in ['dengue', 'S_dengue']}
    from itertools import product
    pairs = list(product(pdict_back['S_dengue'], pdict_back['dengue']))
    comp = pd.DataFrame([], index=adata_av.index)
    adata_avl = np.log2(0.5 + adata_av)
    for p_sd, p_d in pairs:
        lab = (p_sd, p_d)
        compi = adata_avl[p_sd] - adata_avl[p_d]
        comp[lab] = compi
    fr_pos = (comp > 0).mean(axis=1)
    fr_neg = (comp < 0).mean(axis=1)

    gene = 'ENO1'
    datap = adata_av.loc[[gene], pdict.index].T
    datap['Condition'] = pdict.loc[datap.index]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.swarmplot(
        data=datap,
        x='Condition', y=gene,
        palette=cmap,
        order=['dengue', 'S_dengue'],
        alpha=0.8,
        )
    ax.set_ylabel('Gene exp\n[cptt]')
    ax.set_title(gene)
    ax.grid(True)
    ax.set_xticklabels(['nonprog', 'prog'], rotation=90)
    ax.set_xlabel('')
    fig.tight_layout()
    fig.savefig(fig_fdn+'gene_exp_mono_ENO1.png', dpi=300)

    gene = 'CLECL1'
    datap = adata_av.loc[[gene], pdict.index].T
    datap['Condition'] = pdict.loc[datap.index]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.swarmplot(
        data=datap,
        x='Condition', y=gene,
        palette=cmap,
        order=['dengue', 'S_dengue'],
        alpha=0.8,
        )
    ax.set_ylabel('Gene exp\n[cptt]')
    ax.set_title(gene)
    ax.grid(True)
    ax.set_xticklabels(['nonprog', 'prog'], rotation=90)
    ax.set_xlabel('')
    fig.tight_layout()
    fig.savefig(fig_fdn+'gene_exp_mono_CLECL1.png', dpi=300)

    print('ML on monocyte gene exp')
    genes_up = fr_pos.nlargest(10).index.tolist()
    genes_down = fr_neg.nlargest(10).index.tolist()
    genes_pred = genes_up + genes_down
    datab = adata_av.loc[genes_pred].T.loc[pdict.index]
    X = datab.values
    y = (pdict == 'S_dengue').values

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import LeaveOneOut, LeavePOut
    cv_results = cross_validate(
            model, X, y,
            cv=LeavePOut(3),
            scoring=('accuracy', 'precision', 'f1'),
            )

    model.fit(X, y)

    xdata = X @ model.coef_[0]
    xmodel = np.linspace(0, xdata.max() * 1.3, 100)
    ydata = 1 / (1 + np.exp(-xdata - model.intercept_))
    ymodel = 1 / (1 + np.exp(-xmodel - model.intercept_))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(xmodel, ymodel, lw=2, color='grey')
    datap = pdict.to_frame()
    datap['Gene exp\n[combined]'] = xdata
    datap['$P_{severe}$'] = ydata
    sns.scatterplot(
        data=datap, x='Gene exp\n[combined]', y='$P_{severe}$',
        hue='Condition', palette=cmap,
        ax=ax,
        legend=False,
        zorder=10,
        alpha=0.8,
        )
    xcrit = xmodel[((ymodel - 0.5)**2).argmin()]
    ax.axvline(xcrit, color='k', ls='--')
    fig.tight_layout()
    fig.savefig(fig_fdn+'gene_exp_logistic_classifier.png', dpi=300)


    print('Plot HLA-DR/Q genes')
    genes_mhcii = comp.index[comp.index.str.startswith('HLA-D')]
    genes_mhcii = [
        'HLA-DMA',
        'HLA-DMB',
        'HLA-DOA',
        'HLA-DOB',
        'HLA-DPA1',
        'HLA-DPB1',
        'HLA-DQA1',
        'HLA-DQA2',
        'HLA-DQB1',
        'HLA-DQB2',
        'HLA-DRA',
        'HLA-DRB1',
        'HLA-DRB5',
    ]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    datap = 1.0 - fr_neg[genes_mhcii].to_frame(name='y')
    datap = datap.sort_values('y', ascending=False)
    datap['x'] = np.arange(len(genes_mhcii))
    sns.scatterplot(
        data=datap, x='x', y='y', ax=ax,
        hue='y', palette='RdBu', hue_norm=(0, 1),
        s=[100] * len(datap),
        legend=False,
    )
    ax.set_xticks(datap['x'])
    ax.set_xticklabels(datap.index, rotation=90)
    ax.axhline(0.5, ls='--', lw=2, color='k', alpha=0.9)
    ax.set_xlabel('')
    ax.set_ylabel('prog higher\nthan nonprog')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_fdn+'gene_exp_MHC-II.png', dpi=300)


    print('Viral harbouring cells')
    adatadws = adata[adata.obs['Condition'] == 'DWS']
    n_vrna = adatadws.obs.loc[adata.obs['DENV_reads'] > 0, 'cell_subtype_new'].value_counts()
    n_tot = adatadws.obs['cell_subtype_new'].value_counts()
    vhc = pd.concat([n_vrna, n_tot], axis=1).fillna(0)
    vhc.columns = ['n_vhc', 'n_tot']
    vhc.loc['T cell'] = vhc.loc[['CD4+ T cells', 'CD8+ effector T cells', 'CD8+ naive/memory T cells']].sum(axis=0)
    vhc.loc['B cell'] = vhc.loc[['naive B cells', 'cycling Plasmablasts', 'activated B cells', 'memory B cells', 'non_cycling Plasmablasts']].sum(axis=0)
    vhc.loc['NK cell'] = vhc.loc[['XCL_low NK cells', 'XCL_high NK cells']].sum(axis=0)
    vhc.loc['monocyte'] = vhc.loc[['macrophages', 'classical monocytes', 'non_classical monocytes']].sum(axis=0)
    vhc.loc['cDC'] = vhc.loc['conventional DCs']
    vhc.loc['pDC'] = vhc.loc['plasmacytoid DCs']
    vhc['frac'] = vhc['n_vhc'] / vhc['n_tot']

    order = ['T cell', 'NK cell', 'B cell', 'monocyte', 'cDC', 'pDC']
    colors = sns.color_palette(n_colors=7)
    del colors[2]

    fig, ax1 = plt.subplots(figsize=(6.5, 5))
    for ict, ct in enumerate(order):
        y = len(order) - 1 - ict
        x = 100 * vhc.at[ct, 'frac']
        ax1.barh(y, x, left=0, color=colors[ict], zorder=5)
        t1, t2 = int(vhc.at[ct, 'n_vhc']), int(vhc.at[ct, 'n_tot'])
        ax1.text(
            x + 2, y, f'{t1}/{t2}', ha='left', va='center',
            bbox=dict(pad=1, edgecolor=colors[ict], lw=1, facecolor='white'),
        )
    ax1.set_yticks(np.arange(len(order)))
    ax1.set_yticklabels(order[::-1])
    ax1.set_xlabel('% cells with vRNA')
    ax1.set_xlim(0, 65)
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(fig_fdn+'fraction_vhc.png', dpi=300)


    print('Positive/negative strand')
    n_plusminus = adatadws.obs[['DENV_plus', 'DENV_minus', 'DENV_reads']].copy()
    n_plusminus['DENV_plus'] = [int(float(x)) if x != 'nan' else 0 for x in n_plusminus['DENV_plus'].values]
    n_plusminus['DENV_minus'] = [int(float(x)) if x != 'nan' else 0 for x in n_plusminus['DENV_minus'].values]

    fig, ax = plt.subplots(figsize=(5, 3))
    n_plusminus[['DENV_plus', 'DENV_minus']].sum(axis=0).plot.barh(ax=ax, color='grey')
    ax.set_xlabel('# vRNA reads')
    ax.set_yticklabels(['+ strand', '- strand'])
    fig.tight_layout()
    fig.savefig(fig_fdn+'n_vRNA_plus_minus_strand.png', dpi=300)
