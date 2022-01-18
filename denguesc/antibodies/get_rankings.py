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


def pad_sequences(seqs, templates, gap='.'):
    new_seqs = []
    for i, (seq, template) in enumerate(zip(seqs, templates)):
        if seq[0] == gap:
            i_start = len(seq) - len(seq.lstrip(gap))
            seq = template[:i_start] + seq[i_start:]
        if seq[-1] == gap:
            i_end = len(seq.rstrip(gap))
            seq = seq[i_end:] + template[i_end:]
        new_seqs.append(seq)
    return new_seqs


def get_tree_and_rankings(seqs, outgroup_label):
    import subprocess as sp
    import tempfile
    from Bio.Phylo import read as read_tree
    from Bio.SeqIO import parse as parse_seqs

    with tempfile.TemporaryDirectory() as tmpdirname:
        ##if True:
        #tmpdirname = '/home/fabio/Desktop/tmptmp'
        #os.makedirs(tmpdirname, exist_ok=True)

        # Make temporary FASTA file with the sequences
        fn_in = tmpdirname+'/seqs_in.fasta'
        with open(fn_in, 'wt') as fin:
            for name, seq in seqs.items():
                fin.write(f'>{name}\n{seq}\n')

        # Make multiple sequence alignment with MUSCLE
        fn_out = tmpdirname+'/ali_out.fasta'
        sp.run(f'muscle -in {fn_in} -out {fn_out} -diags', shell=True, check=True)

        # Get terminal alignments
        leaves_seqs = list(parse_seqs(fn_out, 'fasta'))

        ## Make phylogenetic tree from MSA
        #fn_tree = tmpdirname+'/tree.nwk'
        #sp.run(f'FastTree -out {fn_tree} {fn_out}', shell=True, check=True)

        # Make phylogeny and infer LBI rankings via Richard's package
        # https://github.com/neherlab/FitnessInference (adapted to Python 3)
        rank_seqs_fdn = '/home/fabio/university/PI/projects/dengue/libraries/FitnessInference'
        rank_seqs_bin = 'rank_sequences.py'
        tmpdir_infer = tmpdirname+'/fitness_inference'
        os.makedirs(tmpdir_infer, exist_ok=True)
        sp.run(
            f'python {rank_seqs_bin} --aln {fn_out} --outgroup {outgroup_label}',
            shell=True, cwd=rank_seqs_fdn, check=True,
            )
        subfdn = [x for x in os.listdir(rank_seqs_fdn) if x.startswith('2021')][0]
        sp.run(f'mv {rank_seqs_fdn}/{subfdn} {tmpdir_infer}/', shell=True, check=True)
        fn_tree = tmpdir_infer+f'/{subfdn}/reconstructed_tree.nwk'
        fn_anc = tmpdir_infer+f'/{subfdn}/ancestral_sequences.fasta'
        fn_rank_int = tmpdir_infer+f'/{subfdn}/sequence_ranking_nonterminals.txt'
        fn_rank_ter = tmpdir_infer+f'/{subfdn}/sequence_ranking_terminals.txt'

        tree = read_tree(fn_tree, 'newick')
        rank_int = pd.read_csv(fn_rank_int, sep='\t', index_col=0)
        rank_ter = pd.read_csv(fn_rank_ter, sep='\t', index_col=0)
        anc_seqs = list(parse_seqs(fn_anc, 'fasta'))

    return {
        'tree': tree,
        'ranking_internal': rank_int,
        'ranking_leaves': rank_ter,
        'ancestral_sequences': anc_seqs,
        'leaves_sequences': leaves_seqs,
        }


if __name__ == '__main__':

    fdn_data = '../../data/datasets/20200809_20kids/vdj/'
    pnames = os.listdir(fdn_data)
    pnames = [x for x in pnames if os.path.isdir(fdn_data+x)]

    pa = argparse.ArgumentParser()
    pa.add_argument('--patient', choices=pnames, action='append')
    pa.add_argument('--regenerate', action='store_true')
    args = pa.parse_args()

    if args.patient is not None:
        pnames = args.patient

    res_all = {}
    for pname in pnames:
        print(f'Patient {pname}')
        fn_res = f'{fdn_data}/{pname}/ranking_LBI.pkl'
        if args.regenerate or not os.path.isfile(fn_res):
            fn_heavy_sequences = f'../../data/datasets/20200809_20kids/vdj/{pname}/filtered_contig_heavy_germ-pass.tsv'
            if not os.path.isfile(fn_heavy_sequences):
                print('Heavy chain file not found')
                continue
            df = pd.read_csv(fn_heavy_sequences, sep='\t')
            df['patient'] = pname
            df['unique_id'] = pname + '-' + df['sequence_id']

            df['sequence_alignment_pad'] = pad_sequences(
                    df['sequence_alignment'],
                    df['germline_alignment'],
                    )

            #df['germline_ali_aa'] = translate_column(df, 'germline_alignment')
            #df['sequence_ali_pad_aa'] = translate_column(df, 'sequence_alignment_pad')
            #df['sequence_aa'] = translate_column(df, 'sequence_alignment')

            res = {}
            # Top 10 clones per patient
            clone_abundances = df['clone_id'].value_counts()[:10]
            # Also limit to clones with 10+ sequences
            clone_abundances = clone_abundances[clone_abundances >= 5]
            nclones = len(clone_abundances)
            print(f'# clones in this individual: {nclones}')
            for clone_id, df_clone in df.groupby('clone_id'):
                if clone_id not in clone_abundances.index:
                    continue
                print(f'Clone: {clone_id}')

                abu = clone_abundances[clone_id]
                if df_clone['germline_alignment'].value_counts()[0] != abu:
                    print(df_clone['germline_alignment'].value_counts())
                    raise ValueError('Clone has multiple germlnes??')

                seqs = df_clone.set_index('unique_id')['sequence_alignment']
                germ_line = f'germline_clone_{clone_id}'
                seqs.loc[germ_line] = df_clone['germline_alignment'].iloc[0]

                print('Get tree and rankings')
                res_clone = get_tree_and_rankings(seqs, outgroup_label=germ_line)

                res[clone_id] = res_clone

            print(f'Saving results for subject {pname} to file')
            with open(fn_res, 'wb') as fout:
                pickle.dump(res, fout)
        else:
            with open(fn_res, 'rb') as fout:
                res = pickle.load(fout)

        res_all[pname] = res

    print('Saving results for all individuals to file')
    fn_res = '../../data/datasets/20200809_20kids/vdj/ranking_LBI.pkl'
    with open(fn_res, 'wb') as fout:
        pickle.dump(res_all, fout)
