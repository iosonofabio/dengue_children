# vim: fdm=indent
'''
author:     Fabio Zanini
date:       25/01/21
content:    Reconstruct genealogies of large clones and infer rankings
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
    pnames = [x for x in pnames if os.path.isfile(fdn_data+x+'//filtered_contig_heavy_germ-pass.tsv')]

    pa = argparse.ArgumentParser()
    pa.add_argument('--patient', choices=pnames, action='append')
    pa.add_argument('--regenerate', action='store_true')
    args = pa.parse_args()

    if args.patient is not None:
        pnames = args.patient

    res = []
    for pname in pnames:
        print(f'Patient {pname}')
        fn_heavy_sequences = f'../../data/datasets/20200809_20kids/vdj/{pname}/filtered_contig_heavy_germ-pass.tsv'
        df = pd.read_csv(fn_heavy_sequences, sep='\t')
        df['patient'] = pname
        df['unique_id'] = pname + '-' + df['sequence_id']

        resi = df.groupby('clone_id').size().to_frame(name='size')
        resi['patient'] = pname
        res.append(resi)

    res = pd.concat(res)

    pnames_order = res.groupby('patient').sum()['size'].sort_values(ascending=False).index
    cmap = dict(zip(pnames_order, sns.color_palette('husl', n_colors=len(pnames))))
    fig, ax = plt.subplots(figsize=(8, 4))
    for pname in pnames_order:
        color = cmap[pname]
        x = res.groupby('patient').get_group(pname)['size'].sort_values()
        y = 1.0 - np.linspace(0, 1, len(x)) + 0.0001
        nseq = x.sum()
        pcond = get_condition(pname)
        label = f'{pname} ({pcond}): {nseq}'
        ax.plot(x, y, label=label, color=color, lw=2)
    ax.grid(True)
    ax.set_xlabel('Clone size')
    ax.set_ylabel('Fraction clones > x')
    ax.legend(
            loc='upper left', ncol=2,
            bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,
            title='Patient & # clones:',
            )
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()

    pnamesd = {
        'sick': [p for p in pnames_order if get_condition(p) != 'Healthy'],
        'healthy': [p for p in pnames_order if get_condition(p) == 'Healthy'],
        }
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
    for ax, pcat in zip(axs, ['sick', 'healthy']):
        pnames_orderi = pnamesd[pcat]
        for pname in pnames_orderi:
            color = cmap[pname]
            x = res.groupby('patient').get_group(pname)['size'].sort_values()
            y = 1.0 - np.linspace(0, 1, len(x)) + 0.0001
            nseq = x.sum()
            pcond = get_condition(pname)
            label = f'{pname} ({pcond}): {nseq}'
            ax.plot(x, y, label=label, color=color, lw=2)
        ax.grid(True)
        if ax == axs[-1]:
            ax.set_xlabel('Clone size')
        ax.set_ylabel('Fraction clones > x')
        ax.legend(
                loc='upper left', ncol=2,
                bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,
                title='Patient & # clones:',
                )
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig.tight_layout()
