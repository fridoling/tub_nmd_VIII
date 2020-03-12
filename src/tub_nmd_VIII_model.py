#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
import sys
from itertools import product 

class Cell:
    def __init__(self, fitness, mut_prob, tub=0, nmd=1, VIII=0):
        self.genotype = np.array([tub, nmd, VIII])
        self.fitness = fitness[self.genotype[0], self.genotype[1], self.genotype[2]]
    def mutate(self, fitness, mut_prob):
        cell_mutated = copy.deepcopy(self)
        rand_vec = np.random.rand(3)
        mut_prob_gt = np.choose(cell_mutated.genotype, mut_prob)
        to_mutate = mut_prob_gt>rand_vec
        cell_mutated.genotype = cell_mutated.genotype^to_mutate
        cell_mutated.fitness = fitness[cell_mutated.genotype[0], cell_mutated.genotype[1], cell_mutated.genotype[2]]
        return(cell_mutated)
class Population:
    def __init__(self, fitness, mut_prob, size=100, **kwargs):
        self.cells = []
        self.size = size
        for i in range(size):
            self.cells.append(Cell(fitness, mut_prob, **kwargs))
        self.fitness = fitness
        self.mut_prob = mut_prob
    def set_cells(self, cells):
        self.cells = cells
        self.size = len(cells)
    def propagate(self):
        # mutate mother population
        pop_mut = copy.deepcopy(self)
        fitness = self.fitness
        mut_prob = self.mut_prob
        mut_cells = []
        for cell in self.cells:
            mut_cells.append(cell.mutate(fitness, mut_prob))
        # create empty daughter population
        pop_out = Population(fitness, mut_prob, size=0)
        # vector of probabilities for the genotype of the daughters according to fitnesses
        fit_vec = [cell.fitness for cell in mut_cells]
        fit_vec_norm = fit_vec/np.sum(fit_vec)
        # choose daughters among mutated mother population with weighted probabilities
        daughters = np.random.choice(mut_cells, size=len(mut_cells), replace=True, p=fit_vec_norm)
        pop_out.set_cells(daughters)
        return(pop_out)                        
    def get_gene_freqs(self):
        gene_freqs = np.sum([cell.genotype for cell in self.cells], axis=0)
        return(gene_freqs/self.size)
    def get_genotype_freq(self, gt):
        genotype_array = np.array([cell.genotype for cell in self.cells])
        genotype_freq = np.count_nonzero(np.all(genotype_array==gt, axis=1))/self.size
        return(genotype_freq)    
    def get_genotype_freqs(self):
        genotype_array = np.array([cell.genotype for cell in self.cells])
        labels = map(''.join, product('01', repeat=3))
        genotypes = product((0, 1), repeat=3)
        genotype_freqs = {}
        for l,gt in zip(labels, genotypes):
            genotype_freqs[l] = np.count_nonzero(np.all(genotype_array==gt, axis=1))/self.size
        return(genotype_freqs)
    def get_mean_fitness(self):
        mean_fitness = np.mean([cell.fitness for cell in self.cells])
        return(mean_fitness)

class Experiment:
    def __init__(self, fitness, mut_prob, size=100, generations=1, **kwargs):
        self.pop = Population(fitness, mut_prob, size=size, **kwargs)
        self.genotype_freqs = {}
        labels = map(''.join, product('01', repeat=3))
        self.genotypes = list(product([0,1], repeat=3))
        self.labels = list(map(''.join, product('01', repeat=3)))
        self.genotype_freqs = np.zeros((generations,8))
        self.gene_freqs = np.zeros((generations, 3))
        self.mean_fitness = np.zeros(generations)
        self.fitness = fitness
        self.mut_prob = mut_prob
        self.size = size
        self.generations = generations
    def run(self):
        for i in progressbar(range(self.generations), "Computing: ", 40):            
            for j, gt in zip(range(8), self.genotypes):
                self.genotype_freqs[i,j] = self.pop.get_genotype_freq(gt)
            self.gene_freqs[i,] = self.pop.get_gene_freqs()
            self.mean_fitness[i] = self.pop.get_mean_fitness()
            self.pop = self.pop.propagate()
        self.pop = None

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
