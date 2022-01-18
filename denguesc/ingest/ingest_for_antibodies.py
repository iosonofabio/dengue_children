# vim: fdm=indent
'''
author:     Fabio Zanini
date:       23/02/21
content:    Explore plotting features on top of the antibody trees for large
            lineages
'''
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import Bio

import anndata


default_genes = [
        'IGHM',

]


pdic = {
    '1_002_01': '1_002',
    '1_019_01': '1_019',
    '1_075_01': '1_075',
    '1_140_01': '1_140',
    '1_144_01': '1_144',
    '3_012_01': '3_012_01',
    '3_037_01': '3_037_01',
    '3_047_01': '3_047_01',
    '3_074_01': '3_074_01',
    '5_030_01': '5_030',
    '5_041_01': '5_041',
    '5_044_01': '5_044',
    '5_089_01': '5_089',
    '5_154_01': '5_154',
    '5_193_01': '5_193',
    '6_001_01': '6_001',
    '6_020_01': '6_020',
    '6_023_01': '6_023',
    '6_025_01': '6_025',
    '6_028_01': '6_028',
    '1_113_01': '1_113',
    '5_010_01': '5_010',
    '5_130_01': '5_130',
    '6_029_01': '6_029',
}


def pconv_fun(pname):
    pconv = pname.replace('-', '_')
    if not pconv.endswith('_01'):
        pconv = pconv+'_01'
    return pconv


def normalise_patient_names(data):
    # Tree dic or antibody meta
    if isinstance(data, dict):
        # Rename dic keys
        pnames = list(data.keys())
        for pname in pnames:
            pconv = pconv_fun(pname)
            data[pconv] = data.pop(pname)

    else:
        pnames = data['patient'].unique()
        pdic = {x: pconv_fun(x) for x in pnames}
        data['patient'] = data['patient'].replace(pdic)


def get_condition(pname):
    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    pcond = pd.read_csv(
        fdn_data+'../../20201002_merged/patient_conditions.tsv',
        sep='\t',
        index_col=0,
        squeeze=True)
    pconv = pconv_fun(pname)

    return pcond[pconv]


def get_antibody_sequences(pnames=None, clone_ids=None, chain_type='heavy'):
    chaind = {
        'heavy': 'filtered_contig_heavy_germ-pass.tsv',
        'light': 'filtered_contig_light_productive-T.tsv',
    }
    fn = chaind[chain_type]

    fdn_datas = [
        '../../data/datasets/20200809_20kids/vdj/',
        '../../data/datasets/20211005_4morekids/vdj/',
        ]
    pdatasets = {}
    for fdn_data in fdn_datas:
        tmp = os.listdir(fdn_data)
        tmp = [x for x in tmp if os.path.isfile(f'{fdn_data}{x}/{fn}')]
        for pname in tmp:
            pdatasets[pname] = fdn_data

    if pnames is None:
        pnames = list(pdatasets.keys())

    dfs = []
    for pname in pnames:
        fdn_data = pdatasets[pname]
        fn_heavy_sequences = f'{fdn_data}{pname}/{fn}'
        df = pd.read_csv(fn_heavy_sequences, sep='\t')

        if clone_ids is not None:
            df = df.loc[df['clone_id'].isin(clone_ids)]

        df['patient'] = pname
        df['unique_id'] = pname + '-' + df['sequence_id']
        df.set_index('unique_id', inplace=True)
        dfs.append(df)
    dfs = pd.concat(dfs)

    normalise_patient_names(dfs)
    # The cell names are a weird mix... so ad hoc dic
    dfs['cell_id_unique'] = dfs['cell_id'] + '-' + dfs['patient'].replace(pdic)

    return dfs


def get_LBI_trees():
    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    fn_lbi = fdn_data+'ranking_LBI.pkl'
    with open(fn_lbi, 'rb') as f:
        data = pickle.load(f)

    return data


def add_column_from_ge(seq_metac, adata, colnames, defaults):
    cell_id_int = np.intersect1d(seq_metac['cell_id_unique'].unique(), adata.obs_names)
    adatac = adata[cell_id_int]

    for colname, default in zip(colnames, defaults):
        if colname in adatac.var_names:
            col = adatac[:, colname].X
            if not isinstance(col, np.ndarray):
                col = np.asarray(col.todense())[:, 0]
            col = pd.Series(col, index=adatac.obs_names, name=colname).to_dict()
            colnew = []
            for abid, row in seq_metac.iterrows():
                colnew.append(col.get(row['cell_id_unique'], default))
        else:
            col = adatac.obs[colname].to_dict()
            colnew = []
            for abid, row in seq_metac.iterrows():
                colnew.append(col.get(row['cell_id_unique'], default))
        seq_metac[colname] = colnew


def add_node_attribute(tree, meta, colname, default, ancestral=False, log=False):
    from collections import Counter

    col = meta[colname].to_dict()
    for x in tree.find_clades(order='postorder'):
        if x.is_terminal():
            if x.name in col:
                val = col[x.name]
                if log:
                    val = np.log(val)
            else:
                val = default
            setattr(x, colname, val)
        else:
            if ancestral is False:
                setattr(x, colname, default)
            elif ancestral == 'majority':
                vals = [getattr(c, colname) for c in x.clades]
                cou = Counter(vals)
                if len(cou) == 1:
                    setattr(x, colname, vals[0])
                else:
                    # FIXME: ties?
                    setattr(x, colname, cou.most_common(2)[0][0])
            elif ancestral == 'mean':
                vals = [getattr(c, colname) for c in x.clades]
                setattr(x, colname, np.mean(vals))



def load_gene_expression(kind='all', cellnames=None):
    fdn = '../../data/datasets/20201002_merged'
    fnd = {
        'highq': f'{fdn}/mergedata_20200930_high_quality.h5ad',
        'cells_with_antibody': '../../data/datasets/20200809_20kids/cells_with_antibody_sequences.h5ad',
        'all': f'{fdn}/mergedata_20200930.loom',
    }
    if kind != 'all':
        adata = anndata.read(fnd[kind])
        if cellnames is not None:
            cell_meta = adata.obs
            cellnames = np.intersect1d(cellnames, cell_meta.index)
            adata = adata[cellnames]

    else:
        # Manual parsing, else it kills the RAM
        import loompy
        with loompy.connect(fnd[kind]) as dsl:
            # Cell metadata
            cell_meta = {key: val for key, val in dsl.ca.items()}
            cell_meta = pd.DataFrame(cell_meta).set_index('obs_names')

            if cellnames is not None:
                cellnames = np.intersect1d(cellnames, cell_meta.index)
                tmp = pd.Series(np.arange(len(cell_meta)), index=cell_meta.index)
                indices_cells = tmp.loc[cellnames].values
                indices_cells.sort()
                cell_meta = cell_meta.iloc[indices_cells]
                ncells = len(cell_meta)
                print(f'Selecting {ncells} with both antibody and GE data')
            else:
                indices_cells = None

            # Gene metadata
            features = pd.Index(dsl.ra['var_names'], name='GeneName')
            gene_meta = pd.DataFrame([], index=features)
            for col in dsl.ra:
                if col != 'var_names':
                    gene_meta[col] = dsl.ra[col]

            # Counts data
            X = np.zeros((len(cell_meta), len(features)), np.float32)
            if indices_cells is not None:
                left = list(indices_cells)
                blk = 500
                ii = 0
                while left:
                    print(ii, end='\r')
                    batch = left[:blk]
                    if len(left) > blk:
                        left = left[blk:]
                    else:
                        left = []
                    X[ii: ii+blk] = dsl[:, batch].astype(np.float32).T
                    ii += len(batch)
                    #X[ii, :] = dsl[:, i].astype(np.float32)
                print()
            else:
                X[:, :] = dsl[:, :].astype(np.float32)

        adata = anndata.AnnData(
                X=X,
                obs=cell_meta,
                var=gene_meta,
        )

    adata.obs.rename(columns={'ID': 'patient'}, inplace=True)

    # Use the newer cell_meta by Zhiyuan
    cellnames = adata.obs_names
    obs_new = pd.read_csv(
            f'{fdn}/mergedata_20210304_obs.tsv', sep='\t', index_col=0,
            )
    # Align cells
    obs_new = obs_new.loc[cellnames]
    for col in obs_new.columns:
        adata.obs[col] = obs_new[col]

    return adata


def load_all(reprocess=False):
    res = {}

    print('Load sequences and isotypes')
    res['seqdata'] = seq_meta = get_antibody_sequences()
    res['seqdata_light'] = seq_meta_light = get_antibody_sequences(
        chain_type='light',
    )

    print('Load gene expression and metadata')
    if reprocess:
        adata = load_gene_expression(
                kind='all',
                cellnames=seq_meta['cell_id_unique'].values,
                )
    else:
        adata = load_gene_expression(
                kind='cells_with_antibody',
                )

    print('Load trees and LBI')
    gdata = get_LBI_trees()
    normalise_patient_names(gdata)

    pnames = list(gdata.keys())
    conditions = {x: get_condition(x) for x in pnames}

    return {
        'adata': adata,
        'gdata': gdata,
        'seqdata': seq_meta,
        'seqdata_light': seq_meta_light,
        'pnames': pnames,
        'conditions': conditions
    }


# Test ingestion
if __name__ == '__main__':

    #adata = load_gene_expression()
    anti_data = load_all()
    seqdata = anti_data['seqdata']

    adata = anti_data['adata']
