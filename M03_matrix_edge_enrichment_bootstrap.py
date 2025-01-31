__author__ = 'james'

# script to go through the network file and count up edges in humannet by threshold


import argparse
import gzip
import csv
import itertools
import os

from collections import defaultdict

import numpy as np

def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def proc_file(input_file, edges, d0_genes, d1_genes, amax, i, output_dir, n_boot):
    thresholds = np.arange(0, amax, i)

    if amax < 0.0:
        ## Take an edge list and a threshold, sum number less than threshold, but t is a 1D array (?)
        f_e = lambda e,t: (e[...,np.newaxis] <= t).sum(0) ## newaxis adds another dimension to the matrix.
        output_file = os.path.join(output_dir,
                                   os.path.splitext(os.path.basename(input_file))[0] + '_negative_enrichment.txt')
    else:
        f_e = lambda e,t: (e[...,np.newaxis] >= t).sum(0)
        output_file = os.path.join(output_dir,
                                   os.path.splitext(os.path.basename(input_file))[0] + '_enrichment.txt')

    if os.path.exists(output_file):
        return

    n_d0 = sum(len(v) for v in d0_genes.values())
    n_d1 = sum(len(v) for v in d1_genes.values())
    input_data = np.fromfile(input_file, dtype=np.float64).reshape((n_d0, n_d1)) ## throws a cannot reshape error
    ## this line could be problematic. Since not all genes make it through to the input file, which is the Corr. mat

    d0_egenes = {g0 for g0,g1 in edges}
    d1_egenes = {g1 for g0,g1 in edges}

    len_te = 0
    total_counts = np.zeros_like(thresholds)

    for glist in grouper(((g0,g1) for g0,g1 in itertools.product(d0_genes, d1_genes)
                          if g0 != g1 and d0_genes[g0] & d0_egenes and d1_genes[g1] & d1_egenes),
                         1000):
        glist = filter(None, glist)
        if not glist: ## If alll the entries are zero 
            continue

        len_te += len(glist)
        t_ix0,t_ix1 = zip(*[(i0,i1) for g0,g1 in glist for i0,i1
                            in itertools.product(d0_genes[g0], d1_genes[g1])])

        total_counts += f_e(input_data[t_ix0, t_ix1], thresholds)


    # convert to list for bootstrapping
    edgelist0 = np.array(list(edges), dtype=int)
    print edgelist0.shape

    # background rate: [# of actual network edges] / [# possible edges]
    bkrd = float(len(edges)) / len_te

    counts_boot = np.zeros((3, thresholds.shape[0]))
    enrich_boot = np.zeros((3, thresholds.shape[0]))

    enrich_mean = np.zeros(thresholds.shape[0])
    enrich_std = np.zeros(thresholds.shape[0])

    assert thresholds.shape[0] % 10 == 0

    ## memory error here , need ~ 100gb or more depending.
    das_boot = np.random.randint(0, len(edgelist0), (n_boot, len(edgelist0))) ## A matrix, rows are resamples, columns are edges 
    ## nboot x len(edgelest) --> 100,000 x 60,193 --> 48.2 gb memory 

    for i in range(0, thresholds.shape[0], 10): ## step by 10s, i.e in batches of 10
        # counts: number of network edges above each threshold
        ## each row is a bootstrap. Number of edges about each threshold. 
        counts = np.zeros((n_boot, 10), dtype=int) ## counts --> 10 columns, n_bootstraps, 10 at a time
        enrichment = np.zeros_like(counts, dtype=float) ## equivalent number of zeroes

        # bootstrap n_boot different edge lists
        for n in range(n_boot):  ## for each bootstrap
            edgeboot = das_boot[n, :] ## get the edge boot list
            

            ## apply the counting function
            counts[n,:] = f_e(input_data[edgelist0[edgeboot, 0], edgelist0[edgeboot, 1]],
                              thresholds[i:i+10]).astype(float)


        ## for each threshold 
        ## colate the bootstrap results into distributions
        counts_boot[:, i:i+10] = np.vstack(np.percentile(counts, (2.5, 50.0, 97.5), 0)) ## stack into matrix for eahc booth


       
        tci = total_counts[i:i+10] > 0 ## generate an index based on total_counts: same dimensions as threshold aka
        ## an arange --> it's all zeroes when its generated though "zero_like"


        enrichment[:, tci] = (counts[:, tci] / total_counts[i:i+10][tci]) / bkrd ## calculate enrichment by dividing by background

        enrich_boot[:, i:i+10] = np.vstack(np.percentile(enrichment, (2.5, 50.0, 97.5), 0)) ## median is 50% percentile
        enrich_mean[i:i+10] = enrichment.mean(0)
        enrich_std[i:i+10] = enrichment.std(0)

    with open(output_file, 'w') as OUT:
        print >> OUT, '\t'.join(('Bin', 'Total',
                                 'Hnet_lo', 'HNet_median', 'Hnet_hi',
                                 'Enrichment_mu', 'Enrichment_std',
                                 'Enrichment_lo', 'Enrichment_median', 'Enrichment_hi'))

        fmt_str = '%g\t%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g'
        for j in range(thresholds.shape[0]):
            print >> OUT, fmt_str % (thresholds[j], total_counts[j],
                                     counts_boot[0,j], counts_boot[1,j], counts_boot[2,j],
                                     enrich_mean[j], enrich_std[j],
                                     enrich_boot[0,j], enrich_boot[1,j], enrich_boot[2,j])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', nargs='+')

    parser.add_argument('--output_dir')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--network')
    parser.add_argument('--network_threshold', type=float, default=2.0)

    parser.add_argument('--i', type=float)
    parser.add_argument('--max', type=float)
    parser.add_argument('--n', type=int, default=100000) ## Number of bootstraps

    parser.add_argument('--j', type=int)

    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()

    if args.max < 0.0 and args.i > 0.0:
        raise ValueError("Can't mix positive index with negative range")

    np.random.seed(args.random_seed)

    input_file = args.input[args.j] ## used to index out inputs by which jobarray is running
    d0,d1 = os.path.basename(input_file).split('_')[0].split('-') ## get data-types for chosen file
    label_files = {os.path.basename(lf)[:-4].split('_')[1]:lf for lf in args.labels} ## dict mapping data-type to label file

    with open(label_files[d0]) as f:
        rdr = csv.reader(f, delimiter='\t')
        rdr.next()
        d0_genes = [row[0] for row in rdr] ## get the first column
        assert d0_genes == sorted(d0_genes)

        d0_genes_d = defaultdict(set) ## genes is a directory, but you can append entries to an existing entry directly 
        ## i.e if d[0]=a:[a,b,c], then d[0].append(f) --> d[0] = a:[a,b,c,f]
        for i,g in enumerate(d0_genes):
            d0_genes_d[g.split('_')[0]].add(i) ## split on underscores i.e if "i" is >1 position only count unique entries
        # d0_genes = {g:i for i,g in enumerate(sorted(d0_genes))}

    if d0 != d1:
        with open(label_files[d1]) as f:
            rdr = csv.reader(f, delimiter='\t')
            rdr.next()
            d1_genes = [row[0] for row in rdr]
            assert d1_genes == sorted(d1_genes)

            d1_genes_d = defaultdict(set)
            for i,g in enumerate(d1_genes):
                d1_genes_d[g.split('_')[0]].add(i)
            # d1_genes = {g:i for i,g in enumerate(sorted(d1_genes))}
    else:
        # d1_genes = d0_genes
        d1_genes_d = d0_genes_d

    edges = set()

    ## operating on network
    with open(args.network) as f:
        rdr = csv.reader(f, delimiter='\t')
        for row in rdr:
            if args.network_threshold == 0.0 or float(row[2]) >= args.network_threshold:
                ## If network above threshold 
                assert row[0] != row[1] ## check for self-linking node.


                if row[0] in d0_genes_d and row[1] in d1_genes_d: ## If the gene in the network is found in the data-set
                    # edges.add((d0_genes[row[0]], d1_genes[row[1]])) ## effectively a intersection filter

                    ## Index the gene dictionary by the gene in the network, get the mapped gene in the data
                    ## get the itertool product of it. and add it to the set as a tuple, returns numbers not characters
                    ## each tuple contains an index into the data matrix
                    edges.update(itertools.product(d0_genes_d[row[0]], d1_genes_d[row[1]]))
                if row[1] in d0_genes_d and row[0] in d1_genes_d: ## do the opposite of the edges are reversed
                    # edges.add((d0_genes[row[1]], d1_genes[row[0]]))
                    edges.update(itertools.product(d0_genes_d[row[1]], d1_genes_d[row[0]]))

    proc_file(input_file, edges, d0_genes_d, d1_genes_d, # d0_genes, d1_genes
              args.max, args.i, args.output_dir, args.n)
