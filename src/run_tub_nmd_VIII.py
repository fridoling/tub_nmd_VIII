#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import shutil
import pickle
from datetime import datetime
import multiprocessing as mp
import pandas as pd


from tub_nmd_VIII_model import * 


res_folder = "./res/"+str(datetime.now()).split('.')[0]+'/'
if not os.path.isdir('res'):
    os.mkdir('res')
os.mkdir(res_folder)

# read fitness data
fitness_cdc20x1 = np.zeros((2,2,2))
fitness_cdc20x4 = np.zeros((2,2,2))

df_fitness = pd.read_csv('../tub_nmd_VIII/src/fitness.csv', dtype={'genotype': str})
df_fitness.set_index('genotype', inplace=True)

for ind, f1, f4 in df_fitness.itertuples():
    fitness_cdc20x1[int(ind[0]), int(ind[1]), int(ind[2])] = f1    
    fitness_cdc20x4[int(ind[0]), int(ind[1]), int(ind[2])] = f4

for ind, f1, f4 in df_fitness.itertuples():
    fitness_cdc20x1[int(ind[0]), int(ind[1]), int(ind[2])] = f1    
    fitness_cdc20x4[int(ind[0]), int(ind[1]), int(ind[2])] = f4

# define mutation rates
mut_prob_cdc20x1 = np.array([[1e-6, 1e-7, 0.001], [1e-7, 1e-5, 0.0001]])*1000
mut_prob_cdc20x4 = np.array([[1e-6, 1e-7, 0.001], [1e-7, 1e-5, 0.0001]])*1000

# define and run experiments
exp_cdc20x1 = Experiment(fitness_cdc20x1, mut_prob_cdc20x1, size=100)
exp_cdc20x4 = Experiment(fitness_cdc20x4, mut_prob_cdc20x4, size=100)

exp_cdc20x1.run(generations=200)
exp_cdc20x4.run(generations=200)

# save data
res = {}
for i,exp,title in zip(range(2), [exp_cdc20x1, exp_cdc20x4], ['Cdc20x1', 'Cdc20x4']):
    res[title] = {}
    res[title]['genotype_freqs'] = exp.get_genotype_freqs()
    res[title]['gene_freqs'] = exp.get_gene_freqs()
    res[title]['mean_fitness'] = exp.get_mean_fitness()

with open('res/simulation_data.pickle', 'wb') as f:
    pickle.dump(res, f)
