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
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('../../'))
from denguesc.ingest.ingest_for_antibodies import load_all
from denguesc.antibodies.explore_rich_plots import plot_composite

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet import Dataset


def filter_BTCR(comp):
    prefixes = [
        'IGHV', 'IGHJ', 'IGHD',
        'IGKV', 'IGKJ', 'IGLV', 'IGLJ',
        'TRAV', 'TRAJ', 'TRBV', 'TRBJ',
        ]
    idx = pd.Series(np.ones(len(comp), bool), index=comp.index)
    for pfx in prefixes:
        idx &= ~idx.index.str.startswith(pfx)
    return comp.loc[idx]


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


if __name__ == '__main__':

    anti_data = load_all()
    adata = anti_data['adata']
    gdata = anti_data['gdata']
    seq_meta = anti_data['seqdata']
    pnames = anti_data['pnames']
    conditions = anti_data['conditions']

    if True:
        print('Expand to a few large lineages')
        clone_sizes = {}
        for pname in pnames:
            gdatap = gdata[pname]
            for clone_id, gdatac in gdatap.items():
                clone_sizes[(pname, clone_id)] = gdatac['ranking_leaves'].shape[0]
        clone_sizes = pd.Series(clone_sizes).to_frame(name='clone_size')
        clone_sizes['condition'] = clone_sizes.index.get_level_values(0).map(conditions)

    if False:
        print('Look at one for tests')
        pname, clone_id = '1_002_01', '1958_20'
        pname, clone_id = '1_075_01',  '488_23'
        pname, clone_id = '6_023_01',   '450_8'
        pname, clone_id = '6_023_01', '1743_71'

        seq_metac = seq_meta.query(
                '(patient == @pname) & (clone_id == @clone_id)').copy()
        cellnames = np.intersect1d(seq_metac['cell_id_unique'], adata.obs_names)
        adatac = adata[cellnames]
        dic = gdata[pname][clone_id]
        tree = dic['tree']

        # Depth and gene expression
        tmp = {
            'leaf_depth': tree.depths(),
            'leaf_layer': tree.depths(unit_branch_lengths=True),
        }
        d = pd.DataFrame(
                -np.ones((len(adatac), 2)),
                index=adatac.obs_names,
                columns=list(tmp.keys()))
        for col in d.columns:
            tmpi = tmp[col]
            for node, val in tmpi.items():
                leaf_name = node.name
                if leaf_name not in seq_metac.index:
                    continue

                cellid = seq_metac.at[leaf_name, 'cell_id_unique']
                if cellid not in adatac.obs_names:
                    continue

                d.at[cellid, col] = val

        ds = Dataset.from_AnnData(adatac)
        for col in d.columns:
            ds.obs[col] = d[col]

        corr = ds.correlation.correlate_features_phenotypes(
                features='all', phenotypes=['leaf_layer', 'leaf_depth'],
                ).fillna(0)

        plot_composite(
            anti_data, pname, clone_id,
            #genes=['IGHV3-20', 'MPPE1', 'LAMP2', 'LBH', 'KARS', 'MCPH1', 'MTREX', 'EXOSC8'],
            genes=['DDX19B', 'IGLV2-34', 'AIM2', 'LRRIQ3', 'TMEM184C'],
        )

        plt.ion(); plt.show()

    if True:
        print('Differential expression for highest LBI subtree')
        pname, clone_id = '1_002_01', '1958_20'
        #pname, clone_id = '1_075_01',  '488_23'
        #pname, clone_id = '6_023_01',   '450_8'
        #pname, clone_id = '6_023_01', '1743_71'

        seq_metac = seq_meta.query(
                '(patient == @pname) & (clone_id == @clone_id)').copy()
        cellnames = np.intersect1d(seq_metac['cell_id_unique'], adata.obs_names)
        adatac = adata[cellnames]
        dic = gdata[pname][clone_id]
        tree = dic['tree']

        fittest_node_name = dic['ranking_internal'].idxmax()['LBI']
        fittest_node = next(tree.find_clades(name=fittest_node_name))
        for sibiling in tree.root.get_path(fittest_node)[-2].clades:
            if sibiling.name != fittest_node_name:
                break
        is_child_sibiling = {x.name: False for x in tree.get_terminals()}
        is_child = {x.name: False for x in tree.get_terminals()}
        for leaf in fittest_node.get_terminals():
            is_child[leaf.name] = True
        for leaf in sibiling.get_terminals():
            is_child_sibiling[leaf.name] = True
        is_child = pd.Series(is_child)
        is_child.index = [seq_metac.at[x, 'cell_id_unique'] for x in is_child.index]
        is_child_sibiling = pd.Series(is_child_sibiling)
        is_child_sibiling.index = [seq_metac.at[x, 'cell_id_unique'] for x in is_child_sibiling.index]

        ds = Dataset.from_AnnData(adatac)
        ds.obs['is_child_of_top'] = is_child.loc[ds.samplenames]
        ds.obs['is_child_of_sibiling'] = is_child_sibiling.loc[ds.samplenames]
        ds.obs['subtree'] = 'other'
        fr = ds.obs['is_child_of_top'].mean()
        print(f'Fraction of children: {fr}')

        if fr < 0.05 or fr > 0.95:
            sys.exit()

        dsp = ds.split('is_child_of_top')
        comp = dsp[True].compare(dsp[False], method='kolmogorov-smirnov-rich')
        compf = filter_BTCR(comp)
        top = list(compf.nlargest(30, 'statistic').index[:4])
        bottom = list(compf.nsmallest(30, 'statistic').index[:4])

        plot_composite(
            anti_data, pname, clone_id,
            #genes=['IGHV3-20', 'MPPE1', 'LAMP2', 'LBH', 'KARS', 'MCPH1', 'MTREX', 'EXOSC8'],
            genes=top + bottom,
        )
