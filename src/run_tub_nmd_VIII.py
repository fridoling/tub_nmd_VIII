#!/usr/bin/env python
# coding: utf-8

import os
import sys
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
with open('src/fitness.pickle', 'rb') as f:
    fitness = pickle.load(f)

# define mutation rates
mut_prob = {}
mut_prob['cdc20x1'] = np.array([[1e-6, 1e-7, 0.001], [1e-7, 1e-5, 0.0001]])
mut_prob['cdc20x4'] = np.array([[1e-6, 1e-7, 0.001], [1e-7, 1e-5, 0.0001]])

# define and run experiments
res = {}
size = np.int(sys.argv[1])
generations = np.int(sys.argv[2])
for label in ['cdc20x1', 'cdc20x4']:
    sys.stdout.write('run experiment '+label+'\n')
    exp = Experiment(fitness[label], mut_prob[label], size=size, generations=generations)
    exp.run()
    res[label] = exp

with open(res_folder+'simulation_data.pickle', 'wb') as f:
    pickle.dump(res, f)
