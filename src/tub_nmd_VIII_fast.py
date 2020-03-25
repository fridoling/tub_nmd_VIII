#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
import sys


def mutate(genotype_in, mut_matrix):
    genotype_out = np.zeros(8)
    for i in range(8):
        rand_vec = np.random.choice(8, size=int(genotype_in[i]), p=mut_matrix[i,:])
        genotype_out+=np.bincount(rand_vec, minlength=8)
    return(genotype_out)

def propagate(genotype_in, fitness):
    genotype_out = np.zeros(8)
    pop_size = genotype_in.sum(dtype=int)
    freq_vec = fitness*genotype_in
    rand_vec = np.random.choice(8, size=pop_size, p=freq_vec/freq_vec.sum())
    genotype_out = np.bincount(rand_vec, minlength=8)
    return(genotype_out)

def get_mean_fitness(gt, fitness):
    return(np.sum(gt*fitness, axis=1))

def get_mean_fitness3(gt, fitness):
    return(np.dot(gt, fitness))

def get_gene_freqs(gt):
    gene_freq = np.zeros((gt.shape[0], 3))
    for i in range(8):
        bin_i = np.binary_repr(i, width=3)
        for j in range(3):
            if bin_i[j] =='1':
                gene_freq[:,j] += gt[:,i]
    return(gene_freq)
   
def get_gene_freqs3(gt):
    gene_freq = np.zeros((gt.shape[0], gt.shape[1], 3))
    for i in range(8):
        bin_i = np.binary_repr(i, width=3)
        for j in range(3):
            if bin_i[j] =='1':
                gene_freq[:,:,j] += gt[:,:,i]
    return(gene_freq)
    
def convert_mut(mp):
    # convert 2x3 mutation matrix into 8x8
    mut_matrix = np.zeros((8,8))
    for i in range(8):
        bin_i = np.binary_repr(i, width=3)
        for j in range(8):
            bin_j = np.binary_repr(j, width=3)        
            p = 1        
            for k in range(3):
                if int(bin_i[k])>int(bin_j[k]):
                    p*=mp[1,k]
                elif int(bin_i[k])<int(bin_j[k]):
                    p*=mp[0,k]
            mut_matrix[i,j] = p
        mut_matrix[i,i] = 2-np.sum(mut_matrix[i,:])
    return(mut_matrix)

def convert_fitness(fitness):
    # convert 2x2x2 fitness matrix to vector
    fitness_vec = np.zeros(8)
    for i in range(8):
        i_bin = np.binary_repr(i, 3)
        j = int(i_bin[0])
        k = int(i_bin[1])
        l = int(i_bin[2])
        fitness_vec[i] = fitness[j,k,l]
    return(fitness_vec)

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
    
def run_simulation_parallel(n, gt_in, params, label):
    numpy.random.seed()
    generations = params['generations']
    gt = np.zeros((generations, 8))
    mut_prob = convert_mut(params['mut_prob'][label])
    fitness = convert_fitness(params['fitness'][label])    
    gt[0,:] = gt_in
    for i in range(1, generations):
        gt_mut = mutate(gt[i-1,:], mut_prob)
        gt[i,:] = propagate(gt_mut, fitness)
    return(gt)