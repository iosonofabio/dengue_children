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


fig_fdn = '../../figures/ozsinglecell2021/'


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
    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    if pnames is None:
        pnames = os.listdir(fdn_data)
        pnames = [x for x in pnames if os.path.isfile(fdn_data+x+'/filtered_contig_heavy_germ-pass.tsv')]

    dfs = []
    for pname in pnames:
        fn_heavy_sequences = f'../../data/datasets/20200809_20kids/vdj/{pname}/filtered_contig_heavy_germ-pass.tsv'
        df = pd.read_csv(fn_heavy_sequences, sep='\t')

        if clone_ids is not None:
            df = df.loc[df['clone_id'].isin(clone_ids)]

        df['patient'] = pname
        df['unique_id'] = pname + '-' + df['sequence_id']
        df.set_index('unique_id', inplace=True)
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs


def get_LBI_trees():
    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    fn_lbi = fdn_data+'ranking_LBI.pkl'
    with open(fn_lbi, 'rb') as f:
        data = pickle.load(f)

    return data


def plot_tree(
        tree,
        ax=None,
        unit_branch_lengths=False,
        style='square',
        orientation='horizontal',
        color_dots_by=None,
        color_dots_log=None,
        cmap_dots=None,
        kwargs_dots=None,
        kwargs_lines=None,
        ):

    def transform_polar(x, y, ylim):
        r = np.asarray(x)
        theta = 2 * np.pi * np.asarray(y) / ylim
        xnew = r * np.cos(theta)
        ynew = r * np.sin(theta)
        return (xnew, ynew)

    cmaps = {
        'isotype': {
            'IGHM': 'brown',
            'IGHG1': 'red',
            'IGHG2': 'orange',
            'IGHG3': 'gold',
            'IGHG4': 'lawngreen',
            'IGHA1': 'steelblue',
            'IGHA2': 'slateblue',
            'IGHE': 'black',
            'IGHD': 'sandybrown',
        },
        'cell_type': {
            'Plasmablasts': 'darkred',
            'B_cells': 'tomato',
            'T_cells': 'steelblue',
            'NK_cells': 'lawngreen',
            'Monocytes': 'gold',
            'unknown': 'grey',
            'doublets': 'black',
        },
    }
    cmaps['c_call'] = cmaps['isotype']

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    leaves = tree.get_terminals()
    n = len(leaves)

    # Set x/y coordinates of dots
    xd = tree.depths(unit_branch_lengths=unit_branch_lengths)
    m = max(xd.values())
    yd = {}
    for il, leaf in enumerate(leaves):
        yd[leaf] = il
    for node in tree.get_nonterminals(order='postorder'):
        y = np.mean([yd[x] for x in node.clades])
        yd[node] = y

    points = np.array([(xd[node], yd[node]) for node in xd])
    x, y = points.T

    # Color strategies for tree nodes
    # NOTE: these mainly rely on node attributes, e.g. node.isotype
    if cmap_dots is None:
        if color_dots_by is None:
            cdots = 'tomato'
        else:
            if color_dots_by in cmaps:
                cmap = cmaps[color_dots_by]
                def color_dot_fun(node):
                    if isinstance(node, Bio.Phylo.Newick.Clade):
                        val = getattr(node, color_dots_by)
                    else:
                        val = node
                    return cmap.get(val, 'grey')
                cdots = [color_dot_fun(node) for node in xd]
                ax._color_dot_fun = color_dot_fun
            else:
                cmap = plt.cm.get_cmap('viridis')
                vals = [getattr(node, color_dots_by) for node in xd]
                if color_dots_log:
                    vals = [np.log(v) for v in vals]
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
                def color_dot_fun(node):
                    if isinstance(node, Bio.Phylo.Newick.Clade):
                        val = getattr(node, color_dots_by)
                    else:
                        val = node
                    if color_dots_log:
                        val = np.log(val)
                    if np.isnan(val):
                        return 'grey'
                    return cmap((val - vmin) / (vmax - vmin))
                cdots = [color_dot_fun(node) for node in xd]
                ax._color_dot_fun = color_dot_fun
    else:
        cdots = [cmap_dots[node] for node in xd]

    kwargs_dots_ = dict(s=30, alpha=0.4, zorder=3)
    if kwargs_dots is not None:
        kwargs_dots_.update(kwargs_dots)

    # Scatter dots
    xs, ys = x, y
    if orientation == 'polar':
        xs, ys = transform_polar(x, y, n)
    ax.scatter(xs, ys, c=cdots, **kwargs_dots_)

    # Square lines
    kwargs_lines_ = dict(lw=1, color='k', alpha=0.7, zorder=2)
    if kwargs_lines is not None:
        kwargs_lines_.update(kwargs_lines)
    for node in tree.get_nonterminals(order='preorder'):
        x0, y0 = xd[node], yd[node]
        if style == 'square':
            for child in node.clades:
                x1, y1 = xd[child], yd[child]
                xs = [x0, x0, x1]
                ys = [y0, y1, y1]
                if orientation == 'polar':
                    xs, ys = transform_polar(xs, ys, n)
                ax.plot(xs, ys, **kwargs_lines_)
        elif style == 'straight':
            for child in node.clades:
                x1, y1 = xd[child], yd[child]
                xs = [x0, x1]
                ys = [y0, y1]
                if orientation == 'polar':
                    xs, ys = transform_polar(xs, ys, n)
                ax.plot(xs, ys, **kwargs_lines_)
        elif style == 'wavy':
            from scipy.interpolate import pchip_interpolate
            if unit_branch_lengths:
                blmin = 1
            else:
                blmin = min([x.branch_length for x in node.clades])
            if blmin > 0:
                x2, x3 = x0 + 0.2 * blmin, x0 + 0.8 * blmin
                xint = np.linspace(x0, x0 + blmin, 10)
            else:
                xint = [x0]
            for child in node.clades:
                x1, y1 = xd[child], yd[child]
                if blmin > 0:
                    yint = pchip_interpolate(
                            [x0, x2, x3, x0 + blmin],
                            [y0, y0, y1, y1], xint)
                else:
                    yint = [y0]
                xs = list(xint) + [x1]
                ys = list(yint) + [y1]
                if orientation == 'polar':
                    xs, ys = transform_polar(xs, ys, n)
                ax.plot(xs, ys, **kwargs_lines_)

    # Set lims
    if orientation == 'horizontal':
        ax.set_ylim(-1, n)
        ax.set_xlim(-0.05 * m, 1.05 * m)
    elif orientation == 'polar':
        ax.set_xlim(-1.02 * m, 1.02 * m)
        ax.set_ylim(-1.02 * m, 1.02 * m)
        ax.set_axis_off()

    return {
        'fig': fig,
        'ax': ax,
    }


def add_column_from_ge(seq_metac, adata, colnames, defaults):
    cell_id_int = np.intersect1d(seq_metac['cell_id_unique'].unique(), adata.obs_names)
    adatac = adata[cell_id_int]

    for colname, default in zip(colnames, defaults):
        if colname in adatac.var_names:
            col = adatac[:, colname].X
            if not isinstance(col, np.ndarray):
                col = np.asarray(col.todense())
            col = col[:, 0]
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


def plot_composite(
        anti_data, pname, clone_id,
        genes=('CD38', 'JCHAIN', 'MKI67'),
    ):

    adata = anti_data['adata']
    gdata = anti_data['gdata']
    seq_meta = anti_data['seqdata']

    seq_metac = seq_meta.query('(patient == @pname) & (clone_id == @clone_id)').copy()
    dic = gdata[pname][clone_id]
    tree = dic['tree']
    rs = pd.concat([dic['ranking_leaves'], dic['ranking_internal']])
    for node in tree.find_clades():
        node.lbi = rs.at[node.name, 'LBI']

    ncols = (3 + len(genes)) // 2 + int(len(genes) % 2 == 0)
    width = 3.8 * ncols
    fig, axs = plt.subplots(2, ncols, figsize=(width, 7.6), sharex=True, sharey=True)
    axs = axs.ravel()

    ax = axs[0]
    plot_tree(
        tree, ax=ax, style='wavy',
        color_dots_by='lbi',
        orientation='polar',
        kwargs_dots=dict(alpha=0.9, s=50),
        kwargs_lines=dict(alpha=0.2),
        )
    ax.text(0.05, 0.95, 'LBI', ha='left', transform=ax.transAxes,
            bbox=dict(pad=4.0, facecolor='w', edgecolor='grey'))

    ax = axs[1]
    add_node_attribute(tree, seq_metac, 'c_call', 'missing', ancestral='majority')
    plot_tree(
        tree, ax=ax, style='wavy',
        color_dots_by='c_call',
        orientation='polar',
        kwargs_dots=dict(alpha=0.9, s=50),
        kwargs_lines=dict(alpha=0.2),
        )
    isotype_seen = set([node.c_call for node in tree.get_terminals()])
    labels = [x if (isinstance(x, str)) or (not np.isnan(x)) else 'N.A.' for x in isotype_seen]
    if 'N.A.' in labels:
        labels.remove('N.A.')
        labels.append('N.A.')
    hs = [ax.scatter([], [], c=ax._color_dot_fun(x)) for x in labels]
    ax.legend(hs, labels)

    ax = axs[2]
    add_column_from_ge(seq_metac, adata, ['cell_type'], ['missing'])
    add_node_attribute(tree, seq_metac, 'cell_type', 'missing', ancestral='majority')
    plot_tree(
        tree, ax=ax, style='wavy',
        color_dots_by='cell_type',
        orientation='polar',
        kwargs_dots=dict(alpha=0.9, s=50),
        kwargs_lines=dict(alpha=0.2),
        )
    labels = np.unique([node.cell_type for node in tree.find_clades()])
    hs = [ax.scatter([], [], c=ax._color_dot_fun(x)) for x in labels]
    ax.legend(hs, labels)

    genes = list(genes)
    add_column_from_ge(seq_metac, adata, genes, [-1] * len(genes))
    seq_metac[genes] += 0.1  # Pseudocounts
    for ig, gene in enumerate(genes):
        add_node_attribute(tree, seq_metac, gene, 0.1, ancestral='mean', log=True)
        ax = axs[3 + ig]
        plot_tree(
            tree, ax=ax, style='wavy',
            color_dots_by=gene,
            orientation='polar',
            kwargs_dots=dict(alpha=0.9, s=50),
            kwargs_lines=dict(alpha=0.2),
            )
        ax.text(0.05, 0.95, gene, ha='left', transform=ax.transAxes,
                bbox=dict(pad=4.0, facecolor='w', edgecolor='grey'))
        if True:
            colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, 20))
            w, h = 0.04, 0.16 / len(colors)
            for ic, color in enumerate(colors[::-1]):
                ax.add_artist(plt.Rectangle(
                    (0.98 - w, 0.97 - h * (ic + 1)), w, h, facecolor=color,
                    edgecolor='none', lw=1,
                    transform=ax.transAxes,
                    zorder=3,
                ))
            ax.arrow(0.98 - 0.5 * w, 0.97 - h * (len(colors) - 0.3),
                     0, h * (len(colors) - 0.6),
                     width=0.003, head_width=0.02, overhang=0.3,
                     length_includes_head=True,
                     transform=ax.transAxes,
                     color='k', alpha=0.6, zorder=5,
                     )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.suptitle(f'{pname}, {clone_id}')

    return {
        'fig': fig,
        'axs': axs,
    }


if __name__ == '__main__':

    anti_data = load_all()
    adata = anti_data['adata']
    gdata = anti_data['gdata']
    seq_meta = anti_data['seqdata']
    pnames = anti_data['pnames']
    conditions = anti_data['conditions']

    sys.exit()

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
        print('Plot all together')
        pname, clone_id = '1_002_01', '1958_20'
        pname, clone_id = '1_075_01',  '488_23'
        pname, clone_id = '6_023_01',   '450_8'
        pname, clone_id = '6_023_01', '1743_71'
        plot_composite(
            anti_data, pname, clone_id,
            genes=['CD27', 'CD38', 'JCHAIN', 'MKI67', 'XBP1', 'IFI27', 'MS4A1'],
            )

    if True:
        pname, clone_id = ('1_075_01', '506_5')
        print(pname, clone_id)
        genes=[
            #'CD27',
            #'CD38',
            #'JCHAIN',
            'MKI67',
            #'XBP1',
            #'IFI27',
            #'MS4A1',
            ]
       

        adata = anti_data['adata']
        gdata = anti_data['gdata']
        seq_meta = anti_data['seqdata']

        seq_metac = seq_meta.query('(patient == @pname) & (clone_id == @clone_id)').copy()
        dic = gdata[pname][clone_id]
        tree = dic['tree']
        rs = pd.concat([dic['ranking_leaves'], dic['ranking_internal']])
        for node in tree.find_clades():
            node.lbi = rs.at[node.name, 'LBI']

        fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True)

        ax = axs[2]
        plot_tree(
            tree, ax=ax, style='wavy',
            color_dots_by='lbi',
            orientation='polar',
            kwargs_dots=dict(alpha=0.9, s=50),
            kwargs_lines=dict(alpha=0.2),
            )
        ax.text(0.05, 0.95, 'LBI', ha='left', transform=ax.transAxes,
                bbox=dict(pad=4.0, facecolor='w', edgecolor='grey'))

        ax = axs[0]
        add_node_attribute(tree, seq_metac, 'c_call', 'missing', ancestral='majority')
        plot_tree(
            tree, ax=ax, style='wavy',
            color_dots_by='c_call',
            orientation='polar',
            kwargs_dots=dict(alpha=0.6, s=50),
            kwargs_lines=dict(alpha=0.2),
            )
        isotype_seen = set([node.c_call for node in tree.get_terminals()])
        labels = [x if (isinstance(x, str)) or (not np.isnan(x)) else 'N.A.' for x in isotype_seen]
        if 'N.A.' in labels:
            labels.remove('N.A.')
            labels.append('N.A.')
        hs = [ax.scatter([], [], c=ax._color_dot_fun(x)) for x in labels]
        ax.legend(hs, labels)

        #ax = axs[1]
        #add_column_from_ge(seq_metac, adata, ['cell_type'], ['missing'])
        #add_node_attribute(tree, seq_metac, 'cell_type', 'missing', ancestral='majority')
        #plot_tree(
        #    tree, ax=ax, style='wavy',
        #    color_dots_by='cell_type',
        #    orientation='polar',
        #    kwargs_dots=dict(alpha=0.9, s=50),
        #    kwargs_lines=dict(alpha=0.2),
        #    )
        #labels = np.unique([node.cell_type for node in tree.find_clades()])
        #hs = [ax.scatter([], [], c=ax._color_dot_fun(x)) for x in labels]
        #ax.legend(hs, labels)

        genes = list(genes)
        add_column_from_ge(seq_metac, adata, genes, [-1] * len(genes))
        seq_metac[genes] += 0.1  # Pseudocounts
        for ig, gene in enumerate(genes):
            add_node_attribute(tree, seq_metac, gene, 0.1, ancestral='mean', log=True)
            ax = axs[1 + ig]
            plot_tree(
                tree, ax=ax, style='wavy',
                color_dots_by=gene,
                orientation='polar',
                kwargs_dots=dict(alpha=0.9, s=50),
                kwargs_lines=dict(alpha=0.2),
                )
            ax.text(0.05, 0.95, gene, ha='left', transform=ax.transAxes,
                    bbox=dict(pad=4.0, facecolor='w', edgecolor='grey'))
            if True:
                colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, 20))
                w, h = 0.04, 0.16 / len(colors)
                for ic, color in enumerate(colors[::-1]):
                    ax.add_artist(plt.Rectangle(
                        (0.98 - w, 0.97 - h * (ic + 1)), w, h, facecolor=color,
                        edgecolor='none', lw=1,
                        transform=ax.transAxes,
                        zorder=3,
                    ))
                ax.arrow(0.98 - 0.5 * w, 0.97 - h * (len(colors) - 0.3),
                         0, h * (len(colors) - 0.6),
                         width=0.003, head_width=0.02, overhang=0.3,
                         length_includes_head=True,
                         transform=ax.transAxes,
                         color='k', alpha=0.6, zorder=5,
                         )

        for ax in axs:
            ax.set_xlim(-0.1092037892023544, 0.1260848844146668)
            ax.set_ylim(-0.16542532095752321, 0.09657239189163885)

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.suptitle(f'{pname}, {clone_id}')

        fig.savefig(f'{fig_fdn}/{pname}_{clone_id}.png', dpi=300)


    plt.ion(); plt.show()

