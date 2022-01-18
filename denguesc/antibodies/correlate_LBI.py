# vim: fdm=indent
'''
author:     Fabio Zanini
date:       27/01/21
content:    Correlate LBI with other features.
'''
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def get_condition(pname):
    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    pcond = pd.read_csv(
        fdn_data+'../../20201002_merged/patient_conditions.tsv',
        sep='\t',
        index_col=0,
        squeeze=True)

    pconv = pname.replace('-', '_')
    if not pconv.endswith('_01'):
        pconv = pconv+'_01'
    return pcond[pconv]


if __name__ == '__main__':

    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    pnames = os.listdir(fdn_data)
    pnames = [x for x in pnames if os.path.isfile(fdn_data+x+'/filtered_contig_heavy_germ-pass.tsv')]

    fn_lbi = fdn_data+'ranking_LBI.pkl'
    with open(fn_lbi, 'rb') as f:
        data = pickle.load(f)

    lineages = []
    for pname in pnames:
        fn_heavy_sequences = f'../../data/datasets/20200809_20kids/vdj/{pname}/filtered_contig_heavy_germ-pass.tsv'
        df = pd.read_csv(fn_heavy_sequences, sep='\t')
        df['patient'] = pname
        df['unique_id'] = pname + '-' + df['sequence_id']
        df.set_index('unique_id', inplace=True)

        datap = data[pname]
        for clone, datac in datap.items():
            ranks = datac['ranking_leaves']
            idx = ranks.index
            ranks['c_call'] = df.loc[idx, 'c_call']

            nbyconst = ranks.groupby('c_call').size().sort_values(ascending=False)
            if len(nbyconst) == 1:
                continue
            if nbyconst.iloc[1] < 10:
                continue

            lineages.append((pname, clone, nbyconst))

    for (pname, clone, _) in lineages:
        fn_heavy_sequences = f'../../data/datasets/20200809_20kids/vdj/{pname}/filtered_contig_heavy_germ-pass.tsv'
        df = pd.read_csv(fn_heavy_sequences, sep='\t')
        df['patient'] = pname
        df['unique_id'] = pname + '-' + df['sequence_id']
        df.set_index('unique_id', inplace=True)

        datac = data[pname][clone]
        ranks = datac['ranking_leaves']
        idx = ranks.index
        ranks['c_call'] = df.loc[idx, 'c_call']
        nseqs = len(ranks)

        isotypes = ranks['c_call'].value_counts().index
        fig, ax = plt.subplots(figsize=(3, 3))
        cmap = {
            'IGHM': 'grey',
            'IGHG1': 'red',
            'IGHG2': 'orange',
            'IGHG3': 'gold',
            'IGHG4': 'lawngreen',
            'IGHA1': 'steelblue',
            'IGHA2': 'slateblue',
            'IGHE': 'black',
            'IGHD': 'silver',
        }
        for iso in isotypes:
            color = cmap[iso]
            x = np.sort(ranks.loc[ranks['c_call'] == iso, 'LBI'])
            y = 1.0 - np.linspace(0, 1, len(x))
            ax.plot(x, y, label=iso, color=color, lw=2)
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_xlabel('LBI')
        ax.set_ylabel('Fraction of sequences\nwith LBI > x')
        ax.set_title(f'{pname}, {clone}, {nseqs}')
        fig.tight_layout()




