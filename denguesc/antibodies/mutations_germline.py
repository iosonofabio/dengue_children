# vim: fdm=indent
'''
author:     Fabio Zanini
date:       25/01/21
content:    Reconstruct genealogies of large clones and infer rankings
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd


def translate_column(df, col):
    from Bio.Seq import translate
    aas = []
    for i, seq in enumerate(df[col]):
        seq = seq.replace('.', '')
        for rf in range(3):
            seq_rf = seq[rf:]
            seq_rf = seq_rf[:(len(seq_rf) // 3) * 3]

            try:
                aa = translate(seq_rf)
            except:
                continue

            n_stops = aa.count('*')
            if n_stops == 0:
                break
        else:
            print(f'WARNING: in sequence {i}, all reading frames have problems!')
            aa = ''

        aas.append(aa)
    return aas

def pad_sequences(seqs, template, gap='.'):
    for i, seq in enumerate(seqs):
        if seq[0] == gap:
            i_start = len(seq) - len(seq.lstrip(gap))
            seq[:i_start] = template[:i_start]
        if seq[-1] == gap:
            i_end = len(seq.rstrip(gap))
            seq[i_end:] = template[i_end:]


if __name__ == '__main__':

    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    pnames = os.listdir(fdn_data)
    pnames = [x for x in pnames if os.path.isdir(fdn_data+x)]

    pa = argparse.ArgumentParser()
    pa.add_argument('--patient', choices=pnames, action='append')
    args = pa.parse_args()

    pnames = args.patient

    for pname in pnames:
        print(f'Patient {pname}')
        fn_heavy_sequences = f'../../data/datasets/20200809_20kids/vdj/{pname}/filtered_contig_heavy_germ-pass.tsv'
        df = pd.read_csv(fn_heavy_sequences, sep='\t')

        df['germline_ali_aa'] = translate_column(df, 'germline_alignment')
        df['sequence_ali_aa'] = translate_column(df, 'sequence_alignment')
        df['sequence_aa'] = translate_column(df, 'sequence_alignment')

        res = {}
        clone_abundances = df['clone_id'].value_counts()[:10]
        for clone_id, df_clone in df.groupby('clone_id'):
            if clone_id not in clone_abundances:
                continue
            print(f'Clone: {clone_id}')

            abu = clone_abundances[clone_id]
            if df_clone['germline_ali_aa'].value_counts()[0] != abu:
                print(df_clone['germline_ali_aa'].value_counts())
                break


