#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import pickle
from datetime import datetime
import multiprocessing as mp


from tub_nmd_VIII_model import * 


res_folder = "./res/"+str(datetime.now()).split('.')[0]+'/'
if not os.path.isdir('res'):
    os.mkdir('res')
os.mkdir(res_folder)

# read fitness data
fitness_cdc20x1 = np.zeros((2,2,2))
fitness_cdc20x4 = np.zeros((2,2,2))

with open('src/fitness.pickle', 'rb') as f:
    fitness = pickle.load(f)

# define mutation rates
mut_prob_cdc20x1 = np.array([[1e-6, 1e-7, 0.001], [1e-7, 1e-5, 0.0001]])*1000
mut_prob_cdc20x4 = np.array([[1e-6, 1e-7, 0.001], [1e-7, 1e-5, 0.0001]])*1000

# define and run experiments
exp_cdc20x1 = Experiment(fitness['cdc20x1'], mut_prob_cdc20x1, size=100, generations=200)
exp_cdc20x4 = Experiment(fitness['cdc20x4'], mut_prob_cdc20x4, size=100, generations=200)

exp_cdc20x1.run()
exp_cdc20x4.run()

# save data
res = {}
for exp,title in zip([exp_cdc20x1, exp_cdc20x4], ['cdc20x1', 'cdc20x4']):
    res[title] = exp

with open(res_folder+'simulation_data.pickle', 'wb') as f:
    pickle.dump(res, f)
