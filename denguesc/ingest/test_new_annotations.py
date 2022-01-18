# vim: fdm=indent
'''
author:     Fabio Zanini
date:       05/03/21
content:    Test the new annotations.
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


if __name__ == '__main__':

    print('Load single cell data')
    fn_h5ad = '../../data/datasets/20201002_merged/mergedata_20200930_high_quality.h5ad'
    adata = anndata.read_h5ad(fn_h5ad)
    adata.obs.rename(columns={'ID': 'patient'}, inplace=True)
