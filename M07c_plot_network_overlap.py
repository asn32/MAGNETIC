# coding: utf-8
## deprecated, probably uses older arguments 
import argparse
import csv
import itertools
import os

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('input')
parser.add_argument('random')
parser.add_argument('--randomx', type=int, default=1)

parser.add_argument('--major', default='cutoff',
                    help='Major category for x axis')
parser.add_argument('--minor', default='power',
                    help='Minor category for x axis')
args = parser.parse_args()


def plot_fold_enrichment_bar(major_x, minor_x, d, r_d,
                             major_lbl, minor_lbl, r_x):
    colors = itertools.cycle(plt.rcParams['axes.color_cycle'])

    bar_off = 0.1 if len(minor_x) > 1 else 0.2
    bar_wd = (0.8 / len(minor_x)) if len(minor_x) > 1 else 0.6

    for j,k in enumerate(minor_x):
        y = np.array([np.exp((np.repeat(np.log(d[i][k]), r_x, 0)
                              - np.log(r_d[i][k] + 1.)).mean())
                      for i in major_x])

        ## the calculation of enrichment -->
        ## Over each run, get the value get the log. get the mean
        ## compare it to the baseline 


        yerr = np.exp([np.percentile(np.repeat(np.log(d[i][k]), r_x, 0)
                                     - np.log(r_d[i][k] + 1.), (5.0, 95.0))
                       for i in major_x]) - y[:,np.newaxis]

        plt.bar(np.arange(len(d)) + bar_off + j * bar_wd, y, bar_wd,
                yerr=np.abs(yerr.T), color=colors.next(), ecolor='gray', label=k)

    plt.xlim((0, len(major_x)))
    plt.xticks(np.arange(len(major_x)) + 0.5, major_x, fontsize='small')
    plt.xlabel(major_lbl.capitalize())

    plt.axhline(y=1.0, linestyle='--', color='gray', alpha=0.7)

    plt.ylabel('Enrichment for Interactions')

    if len(minor_x) > 1:
        plt.legend(title=minor_lbl.capitalize(), loc=0)

    plt.title('Cluster Enrichment Over Random')
    plt.show()



with open(args.input) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

with open(args.random) as f:
    rrows = list(csv.DictReader(f, delimiter='\t'))


d = defaultdict(lambda: defaultdict(list))
r_d = defaultdict(lambda: defaultdict(list))

for row in rows:
    d[row[args.major]][row.get(args.minor, 0.0)].append(float(row['total']))

for row in rrows:
    r_d[row[args.major]][row.get(args.minor, 0.0)].append(float(row['total']))


major_x = sorted(d.keys(), key=float)
minor_x = sorted(reduce(set.union, d.values(), set()), key=float)


d = {k:{k2:np.array(d[k][k2]) for k2 in minor_x} for k in major_x}
r_d = {k:{k2:np.array(r_d[k][k2]) for k2 in minor_x} for k in major_x}

plot_fold_enrichment_bar(major_x, minor_x, d, r_d,
                         args.major, args.minor, args.randomx)