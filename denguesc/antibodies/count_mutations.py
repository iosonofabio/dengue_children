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
