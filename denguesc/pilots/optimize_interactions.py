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


known_interactions = [
    ['B_cells', 'CD40', 'T_cells', 'CD40LG'],
]
known_interactions = pd.DataFrame(
        known_interactions,
        columns=['cell_type1', 'gene_name_a', 'cell_type2', 'gene_name_b'],
        )


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

    print('Restrict to interaction genes')
    genes = np.unique(interactions)
    adatag = adata[:, genes]

    print('Split by cell type, adult and children, and condition')
    from anndata_utils.partition import expressing_fractions
    obs = adatag.obs
    adatag.obs['split_col'] = obs['dataset'] + '+' + obs['Condition'].astype(str) + '+' + obs['cell_type'].astype(str)

    split_cols = ['dataset', 'Condition', 'cell_type']
    fracd = expressing_fractions(adatag, split_cols)

    from collections import defaultdict
    th = 0.10
    cell_types = list(obs['cell_type'].cat.categories)
    res = []
    for col in fracd.columns:
        datas, cond, cell_type1 = col
        for cell_type2 in cell_types:
            col2 = (datas, cond, cell_type2)
            fra = fracd.loc[ga, col].values
            frb = fracd.loc[gb, col2].values
            ind = (fra > th) & (frb > th)
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
                }
                res.append(resi)
    res = pd.DataFrame(res)

    res['frac_sum'] = res['frac1'] + res['frac2']

    # Make it redundant (Yike calls it 'merge')
    res2 = res.loc[res['cell_type1'] != res['cell_type2']].copy()
    res2.rename({
        'cell_type1': 'cell_type2',
        'cell_type2': 'cell_type1',
        'gene_name_a': 'gene_name_b',
        'gene_name_b': 'gene_name_a',
    }, inplace=True)
    res2 = res2[res.columns]

    resr = pd.concat([res, res2], axis=0)
