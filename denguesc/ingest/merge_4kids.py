# vim: fdm=indent
'''
author:     Fabio Zanini
date:       16/03/20
content:    Cell type 4 kids samples
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
from singlet import Dataset, CountsTable, FeatureSheet, SampleSheet, concatenate


if __name__ == '__main__':

    sampletable = pd.read_csv('../ingest/sampletable.tsv', sep='\t', index_col=1)
    sampletable['Sample'] = sampletable.index

    samples = ['3_012_01', '3_019_01', '3_037_01', '6_023_01']

    dss = []
    for sample in samples:
        sample_meta = sampletable.loc[sample]

        fdn_counts = '../../data/experiments/{0}/{1}/'.format(
                sample_meta['Run'], sample_meta['Sample'],
                )

        print('Load data for sample {:}'.format(sample))
        ds = Dataset(
            dataset={
                'path': fdn_counts+sample+'.loom',
                'index_samples': 'CellID',
                'index_features': 'GeneName',
                'bit_precision': 32,
                })
        dss.append(ds)

    print('Concatenate')
    dsa = concatenate(dss)

    print('Write to loom file')
    dsa.to_dataset_file(
        '../../data/datasets/20200313_4kids/20200313_4kids.loom',
        )
