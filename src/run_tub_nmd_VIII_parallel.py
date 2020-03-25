#!/usr/bin/env python
# coding: utf-8
import pickle
import os
import sys
from datetime import datetime
import multiprocessing as mp

from tub_nmd_VIII_fast import * 

if len(sys.argv)>4:
    res_folder = "./res/"+sys.argv[4]+'/'
else:
    res_folder = "./res/"+str(datetime.now()).split('.')[0]+'/'
if not os.path.isdir('res'):
    os.mkdir('res')
if not os.path.isdir(res_folder):
    os.mkdir(res_folder)

# read fitness data
with open('src/fitness.pickle', 'rb') as f:
    fitness = pickle.load(f)

fitness['cdc20x1'][1,0,] = 0    
fitness['cdc20x4'][1,0,] = 0    

# define mutation rates
mut_prob = {}
mut_prob['cdc20x1'] = np.array([[1e-8, 0.0, 0.001], [0.0, 1e-7, 0.0001]])
mut_prob['cdc20x4'] = np.array([[1e-8, 0.0, 0.001], [0.0, 1e-7, 0.0001]])


params = {}
params['mut_prob'] = mut_prob
params['fitness'] = fitness


# define and run experiments
params['pop_size'] = np.int(sys.argv[1])
params['generations'] = np.int(sys.argv[2])
params['repeats'] = np.int(sys.argv[3])

gt_in = np.zeros(8)
ancestor = '010'
ancestor_dec = int(ancestor, 2)
gt_in[ancestor_dec] = params['pop_size']

res = {}
for label in ['cdc20x1', 'cdc20x4']:
    res[label] = {}
    sys.stdout.write('run experiment '+label+'\n')
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(run_simulation_parallel, args=(n, gt_in, params, label)) for n in range(params['repeats'])]
    gt_array = np.stack(results)
    gt_array = gt_array/gt_array.sum(axis=2)[:,:,None]
    gf_array = get_gene_freqs3(gt_array)
    mf_array = np.dot(gt_array, convert_fitness(params['fitness'][label]))
    res[label]['genotype_freqs'] = {}
    res[label]['genotype_freqs']['all'] = gt_array
    res[label]['genotype_freqs']['median'] = np.median(gt_array, axis=0)
    res[label]['genotype_freqs']['upper'] = np.percentile(gt_array, 95, axis=0)
    res[label]['genotype_freqs']['lower'] = np.percentile(gt_array, 5, axis=0)
    res[label]['gene_freqs'] = {}
    res[label]['gene_freqs']['median'] = np.median(gf_array, axis=0)
    res[label]['gene_freqs']['upper'] = np.percentile(gf_array, 95, axis=0)
    res[label]['gene_freqs']['lower'] = np.percentile(gf_array, 5, axis=0)
    res[label]['mean_fitness'] = {}
    res[label]['mean_fitness']['median'] = np.median(mf_array, axis=0)
    res[label]['mean_fitness']['upper'] = np.percentile(mf_array, 95, axis=0)
    res[label]['mean_fitness']['lower'] = np.percentile(mf_array, 5, axis=0)    
    
with open(res_folder+'simulation_data.pickle', 'wb') as f:
    pickle.dump((res, params), f)
