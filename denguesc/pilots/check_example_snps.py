# vim: fdm=indent
'''
author:     Fabio Zanini
date:       05/03/20
content:    Check SNP matrices from: https://github.com/10XGenomics/single-cell-3prime-snp-clustering
'''
import os
import sys
import json
import argparse
import numpy as np
from scipy.io import mmread
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet import Dataset, CountsTable, FeatureSheet, SampleSheet



if __name__ == '__main__':

    pa = argparse.ArgumentParser()
    pa.add_argument('--regenerate', action='store_true')
    args = pa.parse_args()

    dataset = 'example'
    dataset = 'ZY_10010001_headtest'
    dataset = 'ZY_10010001'
    fdnroot = '../../data/demux_based_on_mutations/{:}/'.format(dataset)
    fdntrans = '../../data/experiments/{:}/filtered_feature_bc_matrix/'.format(dataset)
    fdn = '../../data/demux_based_on_mutations/{:}/raw_allele_bc_matrices_mex/'.format(dataset)

    print('Load SNP positions')
    snps = pd.read_csv(fdn+'ref/genes.tsv', header=None)
    snps[3] = [x[:-1] for x in snps[3].values]
    snps.rename(columns={0: 'Chromosome', 1: 'Position', 2: 'ref', 3: 'alt'}, inplace=True)

    if args.regenerate:
        print('Subsample cells and store result to file')
        ref = np.asarray(mmread(fdn+'ref/matrix.mtx').todense()).astype(np.float32)
        alt = np.asarray(mmread(fdn+'alt/matrix.mtx').todense()).astype(np.float32)

        with open(fdnroot+'summary.json') as f:
            summary = json.load(f)
        ide = np.array(summary['model2_call'])

        covt = (ref + alt).sum(axis=0)

        print('Take cells that have most coverage')
        covta = covt.argsort()
        indc = covta[:400]
        print('Also take some cell down the midfield that could be the missing sample')
        indc2 = covta[len(indc):len(covta) // 4]
        np.random.shuffle(indc2)
        indc2 = indc2[:800]
        indc = np.concatenate([indc, indc2])
        alt_sub = alt[:, indc]
        ref_sub = ref[:, indc]
        ide = ide[indc]

        print('Take SNPs that have some diversity')
        indg = (alt_sub > 0).sum(axis=1) >= 2

        print('Add SNPs on the Y chromosome, they are important for discrimination')
        snps = pd.read_csv(fdn+'ref/genes.tsv', header=None)
        snps[3] = [x[:-1] for x in snps[3].values]
        snps.rename(columns={0: 'Chromosome', 1: 'Position', 2: 'ref', 3: 'alt'}, inplace=True)
        for i in snps.index[snps['Chromosome'] == 'Y']:
            indg[i] = True

        indg = indg.nonzero()[0]
        alt_sub = alt_sub[indg]
        ref_sub = ref_sub[indg]

        print('Save subsample to files')
        from scipy.io import mmwrite
        mmwrite(fdn+'ref/matrix_sub.mtx', ref_sub, field='real')
        mmwrite(fdn+'alt/matrix_sub.mtx', alt_sub, field='real')
        with open(fdnroot+'ide_sub.tsv', 'wt') as f:
            for i, idei in enumerate(ide):
                f.write('{:}\t{:}\n'.format(indc[i], idei))
        with open(fdnroot+'indg_sub.tsv', 'wt') as f:
            for i in indg:
                f.write('{:}\n'.format(i))

        print('Exit ipython session and restart without --regenerate')
        sys.exit()

    print('Load subsample of cells')
    ref = np.asarray(mmread(fdn+'ref/matrix_sub.mtx')).astype(np.float32)
    alt = np.asarray(mmread(fdn+'alt/matrix_sub.mtx')).astype(np.float32)
    indc = np.zeros(len(ref[0]), int)
    ide = np.zeros(len(ref[0]), int)
    with open(fdnroot+'ide_sub.tsv', 'rt') as f:
        for i, line in enumerate(f):
            if i < len(ide):
                indc[i] = int(line.split('\t')[0])
                ide[i] = int(line.split('\t')[1].rstrip('\n'))

    indg = np.zeros(len(ref), int)
    with open(fdnroot+'indg_sub.tsv', 'rt') as f:
        for i, line in enumerate(f):
            if i < len(indg):
                indg[i] = int(line.rstrip('\n'))
    snps = snps.iloc[indg]

    print('Load transcriptomes')
    genes = pd.read_csv(fdntrans+'features.tsv.gz', sep='\t', compression='gzip', header=None).values[:, 1]
    cells = pd.read_csv(fdntrans+'barcodes.tsv.gz', sep='\t', compression='gzip', header=None, squeeze=True).values
    counts = np.asarray(mmread(fdntrans+'matrix.mtx.gz').todense().astype(np.float32))

    counts = counts[:, indc]
    cells = cells[indc]

    print('Merge counts for genes')
    genes_new = np.unique(genes)
    counts_new = np.zeros((len(genes_new), len(cells)), np.float32)
    for ig, gene in enumerate(genes_new):
        counts_new[ig] = counts[genes == gene].sum(axis=0)
    counts = counts_new
    genes = genes_new
    counts = pd.DataFrame(
        counts,
        index=genes,
        columns=cells,
        )

    print('Find chromosomes for genes')
    biom = pd.read_csv(fdntrans+'mart_export.tsv', sep='\t', index_col=1, squeeze=True)
    biom = biom[biom.isin([str(i+1) for i in range(22)] + ['X', 'Y', 'MT'])]
    chroms = []
    for gene in genes:
        chroms.append(biom.get(gene, ''))
    chroms = pd.Series(chroms, index=genes, dtype='U10')

    counts_y = counts.loc[chroms == 'Y'].sum(axis=0)

    ds = Dataset(
        counts_table=CountsTable(counts),
        )
    ds.featuresheet['Chromosome'] = chroms
    ds.samplesheet['Sex'] = np.array(['F', 'M'])[(counts_y > 3) + 0]
    ds.counts.normalize(inplace=True)

    print('Call cell types')
    print('Feature selection')
    fns = ds.featurenames
    feashort = np.ones(ds.n_features, bool)
    feashort[fns.str.startswith('IGHV')] = False
    feashort[fns.str.startswith('IGHJ')] = False
    feashort[fns.str.startswith('IGLV')] = False
    feashort[fns.str.startswith('IGLJ')] = False
    feashort[fns.str.startswith('IGKV')] = False
    feashort[fns.str.startswith('IGKJ')] = False
    feashort[fns.str.startswith('MT-')] = False
    feashort[fns.str.startswith('RPL')] = False
    feashort[fns.str.startswith('RPS')] = False
    feashort[fns.str.startswith('barcode')] = False
    fnshort = fns[feashort]
    dss = ds.query_features_by_name(fnshort)
    features = dss.feature_selection.overdispersed_within_groups('Sex', inplace=False)
    dsf = ds.query_features_by_name(features)

    print('PCA')
    dsc = dsf.dimensionality.pca(n_dims=30, robust=False, return_dataset='samples')

    print('tSNE')
    vst = dsc.dimensionality.tsne(perplexity=20)

    if False:
        print('Plot cumulative of #alt detected per cell')
        data = (alt > 0).sum(axis=0)
        x = np.sort(data)
        y = 1.0 - np.linspace(0, 1, len(x))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(x, y, lw=2)
        ax.grid(True)
        ax.set_xlabel('# alt alleles per cell')
        ax.set_ylabel('Fraction of cells with > #alt alleles')
        fig.tight_layout()

        print('Plot cumulatives for number of cells each alt is detected in')
        data = (alt > 0).sum(axis=1)
        x = np.sort(data)
        y = 1.0 - np.linspace(0, 1, len(x))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(x, y, lw=2)
        ax.grid(True)
        ax.set_xlabel('# cell with alt allele')
        ax.set_ylabel('Fraction of alleles with > #cells')
        fig.tight_layout()

    #FIXME
    cov = (alt + ref).astype(np.float32)
    af = alt / (cov + 1e-3)

    print('Try and construct distance matrix manually')
    from scipy.spatial.distance import squareform
    nshared = []
    n = af.shape[1]
    cdis = np.ones((n, n))
    cdis[np.arange(n), np.arange(n)] = 0
    k = 0
    for i in range(n):
        ci = cov[:, i]
        afi = af[:, i]
        aci = 1 * (afi > 0.33) + 1 * (afi > 0.67)
        for j in range(i):
            cj = cov[:, j]
            afj = af[:, j]
            acj = 1 * (afj > 0.33) + 1 * (afj > 0.67)

            ind = (ci > 1) & (cj > 1)
            dth = ((afi[ind] > 0.1) & (afj[ind] > 0.1)).sum()
            eth = ind.sum()
            if eth > 0:
                #cdis[i, j] = 1.0 * dth / eth
                #cdis[i, j] = 1.0 / (dth + 1)
                #cdis[i, j] = 1 - (i == (j + 1))  #FIXME
                #cdis[i, j] = 1 - (dth > 3)
                cdis[i, j] = 1 - ((aci[ind] == acj[ind]).mean() >= 0.7)

            cdis[j, i] = cdis[i, j]
            k += 1

            nshared.append((dth, eth))
    nshared = pd.DataFrame(nshared, columns=['n shared alts', 'n_shared cov'])
    pdis = squareform(cdis)

    print('Graph clustering')
    import igraph as ig
    edges = []
    nedges = np.zeros(n)
    for i in range(n):
        for j in range(i):
            if cdis[i, j] < 0.1:
                edges.append([i, j])
                nedges[i] += 1
                nedges[j] += 1
    print(nedges)
    g = ig.Graph(edges)

    import leidenalg
    opt = leidenalg.Optimiser()
    par = leidenalg.CPMVertexPartition(g, resolution_parameter=2e-1)
    opt.optimise_partition(par)
    labels = np.array(par.membership)
    from collections import Counter, defaultdict
    clabels = Counter(labels)
    cdict = defaultdict(list)
    for i, lab in enumerate(labels):
        cdict[lab].append(i)
    print(clabels)

    ll2 = []
    for lab in sorted(clabels.keys()):
        ll2.extend(cdict[lab])

    distc = cdis[ll2].T[ll2].T
    labelsc = labels[ll2]
    countsyc = counts_y[ll2]

    from sklearn.manifold import TSNE
    model = TSNE(perplexity=30, metric='precomputed')
    vs = model.fit_transform(cdis)

    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 10, 5, 5, 5])
    axs = []
    axs.append(fig.add_subplot(gs[:, 0]))
    axs.append(fig.add_subplot(gs[:, 1], sharey=axs[0]))
    axs.append(fig.add_subplot(gs[0, 2]))
    axs.append(fig.add_subplot(gs[1, 2]))
    axs.append(fig.add_subplot(gs[0, 3], sharey=axs[3], sharex=axs[3]))
    axs.append(fig.add_subplot(gs[1, 3], sharey=axs[3], sharex=axs[3]))
    axs.append(fig.add_subplot(gs[0, 4], sharey=axs[3], sharex=axs[3]))
    axs.append(fig.add_subplot(gs[1, 4], sharey=axs[3], sharex=axs[3]))
    ax = axs[0]
    colors2 = sns.color_palette('husl', n_colors=len(clabels))
    colors2 = colors2[::2] + colors2[1::2]
    for i, lab in enumerate(labelsc):
        ax.add_patch(Rectangle((0, i), 1, 1, color=['pink', 'blue'][int(countsyc[i] > 4)]))
        ax.add_patch(Rectangle((1, i), 1, 1, color=colors2[lab]))
    ax.set_xlim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Y chrom', 'cluster'], rotation=90)
    ax = axs[1]
    sns.heatmap(distc, ax=ax, cbar=True)
    ax.set_xlabel('cell')
    ax.set_yticklabels([])
    ax = axs[2]
    c = [colors2[lab] for lab in labels]
    ax.scatter(vs[:, 0], vs[:, 1], s=20, c=c, alpha=0.7)
    ax.set_title('t-SNE based on SNPs')
    ax = axs[3]
    ax.scatter(vst.values[:, 0], vst.values[:, 1], s=20, c=c, alpha=0.7)
    ax.set_title('t-SNE based on gene expression')
    for ig, gene in enumerate(['LYZ', 'CD3E', 'MS4A1', 'CD14']):
        ax = axs[4 + ig]
        ax.set_title(gene)
        ds.plot.scatter_reduced(
            vst,
            ax=ax,
            color_by=gene,
            color_log=True,
            s=20,
            cmap='viridis',
            )
    fig.tight_layout(w_pad=0.3)

    if True:
        print('Hierarchical clustering')
        from scipy.cluster.hierarchy import linkage, leaves_list
        bex = ds.counts.loc[['barcode5', 'barcode6', 'barcode7', 'barcode8'], cells].T
        bex = 1e3 * (bex.T / (bex.sum(axis=1) + 1e-8)).T
        bex = np.log10(bex + 0.1)

        z = linkage(bex.values, method='average', metric='euclidean')
        z = linkage(pdis, method='average', metric='precomputed')
        ll3 = leaves_list(z)

        distc3 = cdis[ll3].T[ll3].T
        labelsc3 = labels[ll3]
        countsyc3 = counts_y[ll3]
        bex3 = bex.iloc[ll3]

        print('Plot matrix')
        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 10, 2])
        axs = []
        axs.append(fig.add_subplot(gs[:, 0]))
        axs.append(fig.add_subplot(gs[:, 1], sharey=axs[0]))
        axs.append(fig.add_subplot(gs[:, 2], sharey=axs[0]))
        colors2 = sns.color_palette('husl', n_colors=len(clabels))
        colors2 = colors2[::2] + colors2[1::2]
        ax = axs[0]
        for i, lab in enumerate(labelsc3):
            ax.add_patch(Rectangle((0, i), 1, 1, color=['pink', 'blue'][int(countsyc3[i] > 4)]))
            ax.add_patch(Rectangle((1, i), 1, 1, color=colors2[lab]))
        ax.set_xlim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Y chrom', 'cluster'], rotation=90)
        ax = axs[1]
        sns.heatmap(distc3, ax=ax, cbar=True)
        ax.set_xlabel('cell')
        ax.set_yticklabels([])
        ax = axs[2]
        sns.heatmap(bex3, ax=ax, cmap='viridis', vmin=-1, vmax=3, cbar=True)
        ax.set_yticklabels([])
        fig.tight_layout()

    plt.ion();plt.show()



    print('List core memberts of bona fide clusters to serve as references')
    cells = pd.read_csv(fdntrans+'barcodes.tsv.gz', sep='\t', compression='gzip', header=None, squeeze=True).values[indc]
    cell_membership = pd.Series(labels, index=cells, name='Cluster')
    # NOTE: we use n_clusters = 4 as a correction
    n_clusters = 4
    core_members = [[] for i in range(n_clusters)]
    for i in range(n_clusters):
        if (n_clusters == 3) or (i in (1, 2)):
            indci = (labels == i).nonzero()[0]
        elif i == 0:
            indci = np.array(ll2)[labelsc == 0][:50]
        else:
            indci = np.array(ll2)[labelsc == 0][-30:]
        core_members[i] = cells[indci]


    print('Load the rest of the data and cluster it onto these clusters')
    cellsa = pd.Index(pd.read_csv(fdntrans+'barcodes.tsv.gz', sep='\t', compression='gzip', header=None, squeeze=True).values)
    ref = np.asarray(mmread(fdn+'ref/matrix.mtx').todense()).astype(np.float32)
    alt = np.asarray(mmread(fdn+'alt/matrix.mtx').todense()).astype(np.float32)

    if False:
        alt_clusters = np.zeros((len(alt), n_clusters))
        ref_clusters = np.zeros((len(ref), n_clusters))
        for i in range(n_clusters):
            indci = cellsa.isin(core_members[i])
            alt_clusters[:, i] = alt[:, indci].sum(axis=1)
            ref_clusters[:, i] = ref[:, indci].sum(axis=1)
        cov_clusters = alt_clusters + ref_clusters
        af_clusters = 1.0 * alt_clusters / (cov_clusters + 1e-3)
        afc_clusters = 1 * (af_clusters > 0.33) + 1 * (af_clusters > 0.66)

        cla = []
        for ic in range(len(cellsa)):
            if (ic % 1000) == 0:
                print(ic, end='\r')
            covi = ref[:, ic] + alt[:, ic]
            frs = [0, 0, 0]
            afi = 1.0 * alt[:, ic] / covi
            afic = 1 * (afi > 0.33) + 1 * (afi > 0.66)
            for i in range(n_clusters):
                inds = (covi > 0) & (cov_clusters[:, i] > 0)
                fr = (afic[inds] == afc_clusters[inds, i]).mean()
                frs[i] = fr
            cla.append(frs)
        print()
        cla = np.array(cla)
        cla[np.isnan(cla)] = 0

    if False:
        points = (cla.T / (cla.sum(axis=1) + 1e-8)).T

        import ternary
        fig, tax = ternary.figure()
        tax.set_title("Similarity to clusters 1-{:}".format(n_clusters), fontsize=20)
        tax.scatter(points, marker='s', color='k', alpha=0.05)
        tax.ticks(axis='lbr', linewidth=1, multiple=0.50, tick_formats="%.1f")
        tax.gridlines(color="blue", multiple=0.50)

    print('Find consensus assignment among really close cells')
    nnei = 11
    # Test on core member cells
    indic = cellsa.isin(np.concatenate(core_members)).nonzero()[0]
    #indic = np.arange(1000)
    n = len(indic)
    cla = np.zeros((n, n_clusters), int)
    for iic, ic in enumerate(indic):
        if (iic % 10) == 0:
            print(iic, end='\r')
        covi = ref[:, ic] + alt[:, ic]
        afi = 1.0 * alt[:, ic] / covi
        afic = 1 * (afi > 0.33) + 1 * (afi > 0.66)

        tmp = np.zeros((n_clusters, nnei))
        for i in range(n_clusters):
            for j in range(nnei):
                altj = alt[:, cellsa == core_members[i][j]][:, 0]
                refj = ref[:, cellsa == core_members[i][j]][:, 0]
                covj = altj + refj
                afj = 1.0 * altj / covj
                inds = (covi > 0) & (covj > 0)
                fr = (afic[inds] == afj[inds]).mean()
                tmp[i, j] = fr
        tmp = Counter(tmp.argmax(axis=0))
        for i in range(n_clusters):
            cla[iic, i] = tmp[i]
        if ic == n - 1:
            break

    cla = pd.DataFrame(cla, index=cellsa[indic])
    cla['cluster'] = -1
    for cn in cla.index:
        for i in range(n_clusters):
            if cn in core_members[i]:
                cla.at[cn, 'cluster'] = i
                break
    cla['predicted_cluster'] = cla.values[:, :4].argmax(axis=1)
    print((cla['cluster'] == cla['predicted_cluster']).mean())

    print('Use sex info')
    n_sex = 5
    cla['sex'] = ds.samplesheet.loc[cla.index, 'Sex']
    cla.loc[cla['sex'] == 'M', 0] += n_sex
    cla.loc[cla['sex'] == 'M', 3] += n_sex
    cla.loc[cla['sex'] == 'F', 1] += n_sex
    cla.loc[cla['sex'] == 'F', 2] += n_sex
    cla['predicted_cluster_withsex'] = cla.values[:, :4].argmax(axis=1)
    print((cla['cluster'] == cla['predicted_cluster']).mean())

    ind_wrong = cla['cluster'] != cla['predicted_cluster']
    claw = cla.loc[ind_wrong]


    if n_clusters == 3:
        points = (cla.T * 1.0 / cla.sum(axis=1)).T

        import ternary
        fig, tax = ternary.figure()
        tax.set_title("Similarity to clusters 1-{:}".format(n_clusters), fontsize=20)
        tax.scatter(points, marker='s', color='k', alpha=0.05)
        tax.ticks(axis='lbr', linewidth=1, multiple=0.50, tick_formats="%.1f")
        tax.gridlines(color="blue", multiple=0.50)

    sys.exit()

    from sklearn.manifold import TSNE
    model = TSNE(perplexity=30, metric='precomputed')
    vs = model.fit_transform(cdis)

    if False:
        print('Calculate allele frequencies ignoring missing data')
        af = (1.0 * alt / (alt + ref + 1e-3)).astype(np.float32)

        print('Try tsne')
        from sklearn.decomposition import PCA
        model = PCA(n_components=10, whiten=True)
        pcs = model.fit_transform(af.T)

        from sklearn.manifold import TSNE
        model = TSNE(perplexity=30)
        vs = model.fit_transform(pcs)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(vs[:, 0], vs[:, 1], s=20, alpha=0.7)
    ax.set_xlabel('dimension 1')
    ax.set_ylabel('dimension 2')
    fig.tight_layout()

    plt.ion();plt.show()
    sys.exit()


    # Only take highly informative SNPs
    ind = np.sort((af > 1e-1).sum(axis=1))[-50:]
    covi = cov[ind]
    afi = af[ind]

    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list

    # Cluster SNPs
    dist = pdist(afi)
    z = linkage(dist)
    ll = leaves_list(z)

    # Cluster cells
    dist = pdist(afi.T)
    z = linkage(dist)
    ll2 = leaves_list(z)

    covic = covi[ll].T[ll2].T
    afic = afi[ll].T[ll2].T

    print('Plot matrix')
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    sns.heatmap(afic, ax=axs[0], cbar=False)
    sns.heatmap(covic, ax=axs[1], cbar=False)
    axs[0].set_title('Alt allele freq')
    axs[1].set_title('Coverage')
    axs[0].set_ylabel('SNP')
    axs[0].set_xlabel('cell')
    axs[1].set_xlabel('cell')
    fig.tight_layout()

    plt.ion()
    plt.show()
