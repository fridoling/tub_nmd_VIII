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

# create plots
fig,ax = plt.subplots(3,2, figsize=(8,7),sharex=True, sharey='row')
for i,exp,title in zip(range(2), [exp_cdc20x1, exp_cdc20x4], ['Cdc20x1', 'Cdc20x4']):
    exp.plot_genotype_freqs(ax=ax[0,i])
    ax[0,i].set_xlabel(None)
    ax[0,i].set_title(title, fontweight='bold')
    exp.plot_gene_freqs(ax=ax[1,i])
    ax[1,i].set_xlabel(None)
    exp.plot_mean_fitness(ax=ax[2,i])
for i in range(3):
    ax[i,1].set_ylabel(None)
ax[0,1].legend(bbox_to_anchor=(1,1))    
ax[1,1].legend(bbox_to_anchor=(1,1))
plt.tight_layout(h_pad=0, w_pad=0)
plt.savefig(res_folder+'simulation.pdf')

