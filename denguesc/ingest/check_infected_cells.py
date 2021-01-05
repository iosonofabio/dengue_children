# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/03/20
content:    Check which cell types are infected.
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
import loompy
from scipy.io import mmread


os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet import Dataset, CountsTable, FeatureSheet, SampleSheet


if __name__ == '__main__':

    sampletable = pd.read_csv('sampletable.tsv', sep='\t', index_col=1)
    sampletable['Sample'] = sampletable.index

    pa = argparse.ArgumentParser()
    pa.add_argument('--sample', required=True, choices=list(sampletable['Sample']))
    args = pa.parse_args()

    sample_meta = sampletable.loc[args.sample]

    fdn_counts = '../../data/experiments/{0}/{1}/'.format(
            sample_meta['Run'], sample_meta['Sample'],
            )

    print('Load data for sample {:}'.format(args.sample))
    ds = Dataset(
        dataset={
            'path': fdn_counts+args.sample+'.loom',
            'index_samples': 'CellID',
            'index_features': 'GeneName',
            'bit_precision': 32,
            })

    indc = ds.samplenames[ds.counts.loc['DENV2'] > 0]

    genes = ['PTPRC', 'CD3E', 'CD14', 'SPON2', 'FCGR3A', 'CD19', 'MS4A1', 'IGHM', 'IGHD', 'IGHG1', 'IGHG2', 'IGHG3']
    df = ds.counts.loc[genes, indc]
    df.columns = np.arange(len(indc))
    print(df)
