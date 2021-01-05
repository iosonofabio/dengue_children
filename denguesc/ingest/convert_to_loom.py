# vim: fdm=indent
'''
author:     Fabio Zanini
date:       14/03/20
content:    Convert 10X Genomics format to loom
'''
import os
import sys
import argparse
import numpy as np
import pandas as pd
import loompy
from scipy.io import mmread



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

    print('Read cell barcodes')
    cellids = pd.read_csv(
            fdn_counts+'filtered_feature_bc_matrix/barcodes.tsv.gz',
            compression='gzip',
            sep='\t',
            index_col=0,
            squeeze=True,
            header=None,
            ).index
    samplesheet = pd.DataFrame([], index=cellids)
    samplesheet['Sample'] = sample_meta['Sample']
    samplesheet['Run'] = sample_meta['Run']
    samplesheet['Sex'] = sample_meta['Sex']
    samplesheet['Age'] = sample_meta['Age']
    samplesheet['Condition'] = sample_meta['Condition']
    samplesheet['CellID'] = sample_meta['Sample']+'-'+samplesheet.index

    print('Read features')
    featuresheet = pd.read_csv(
            fdn_counts+'filtered_feature_bc_matrix/features.tsv.gz',
            compression='gzip',
            sep='\t',
            index_col=0,
            squeeze=True,
            header=None,
            )
    featuresheet.columns = ['GeneName', 'FeatureType']
    featuresheet['EnsemblID'] = featuresheet.index

    print('Read count table')
    counts_raw = mmread(
            fdn_counts+'filtered_feature_bc_matrix/matrix.mtx.gz',
            ).todense()

    print('Dengue reads')
    print('# total DENV2 reads: {:}'.format(counts_raw[-1].sum()))
    print('# cells with 1+ DENV2 reads: {:}'.format((counts_raw[-1] > 0).sum()))

    print('Merge genes with the same name')
    genes = list(featuresheet['GeneName'].unique())
    genes = [x for x in genes if x != 'DENV2'] + ['DENV2']
    counts = np.zeros((len(genes), len(samplesheet)), dtype=np.float32)
    for i, gene in enumerate(genes):
        if (i % 100) == 0:
            print(i, end='\r')
        ind = featuresheet['GeneName'] == gene
        counts[i] += np.asarray(counts_raw[ind]).sum(axis=0)
    print()

    print('Write output file')
    out_fn = fdn_counts+args.sample+'.loom'
    row_attrs = {'GeneName': np.array(genes)}
    col_attrs = {key: samplesheet[key].values for key in samplesheet.columns}
    loompy.create(
            out_fn,
            layers={'': counts},
            row_attrs=row_attrs,
            col_attrs=col_attrs,
            )
