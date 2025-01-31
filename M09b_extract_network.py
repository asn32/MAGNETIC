# coding: utf-8
import argparse
import csv
import os
import itertools

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input', nargs='+')
    parser.add_argument('--labels', nargs='+')

    parser.add_argument('--modules')
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    ## issue here is that label files are not being loaded correctly
    ## used for loading and getting info.
    ## replace with working label file reader snippet from M04
    labels = dict()
    for label_file in args.labels:
        d1 = os.path.basename(label_file[:-4]).split('_')[1] ## get the data-type
        with open(label_file) as f:
            rdr = csv.reader(f, delimiter='\t')
            rdr.next()
            labels[d1] = {r[0]:i for i,r in enumerate(rdr)}

    label_u = sorted(reduce(set.union, labels.values(), set()))

    mmaps = dict()
    for input_file in args.input:
        d1,d2 = os.path.basename(input_file).split('_')[0].split('-')

        mmaps[(d1,d2)] = np.memmap(input_file, dtype=np.float64,
                                   mode='r', shape=(len(labels[d1]), len(labels[d2]))) ## potential problem here too

    dtypes = sorted(reduce(set.union, mmaps, set()))

    with open(args.modules) as f:
        rdr = csv.reader(f, delimiter='\t')
        modules = [row[1:] for row in rdr] ## get the module information


    for ci,module in enumerate(modules):
        output_edges = []

        for g1,g2 in itertools.combinations(module, 2):
            for d1,d2 in itertools.product(dtypes, dtypes):
                if g1 in labels[d1] and g2 in labels[d2]:
                    if d1 <= d2:
                        e = mmaps[(d1,d2)][labels[d1][g1], labels[d2][g2]]
                        if e > args.cutoff:
                            output_edges.append((g1, g2, e, '{}-{}'.format(d1,d2)))
                    else:
                        e = mmaps[(d2,d1)][labels[d2][g2], labels[d1][g1]]
                        if e > args.cutoff:
                            output_edges.append((g2, g1, e, '{}-{}'.format(d2,d1)))

        ## only print-out networks for modules that have a real edge between them above cutoff
        if len(output_edges) > 0:
            with open(os.path.join(args.output_dir, 'module_%s.txt' % ci), 'w') as OUT:
                print >> OUT, '\n'.join('%s\t%s\t%f\t%s' % e for e in output_edges)
