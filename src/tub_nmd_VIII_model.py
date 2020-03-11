#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy

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
    def __init__(self, fitness, mut_prob, size=100, **kwargs):
        self.pops = []
        self.pops.append(Population(fitness, mut_prob, size=size, **kwargs))
        self.fitness = fitness
        self.mut_prob = mut_prob
    def run(self, generations=1, **kwargs):
        fitness = self.fitness
        mut_prob = self.mut_prob
        for i in range(generations):
            self.pops.append(self.pops[-1].propagate())
    def get_gene_freqs(self):
        freq_array = np.zeros((len(self.pops), 3))
        for i in range(len(self.pops)):
            freq_array[i,] = self.pops[i].get_gene_freqs()
        return(freq_array)
    def get_genotype_freqs(self):
        labels = map(''.join, product('01', repeat=3))
        genotypes = product([0,1], repeat=3)
        genotype_freqs = {}
        for l, gt in zip(labels, genotypes):
            freqs_gt = np.zeros(len(self.pops))
            for i in range(len(self.pops)):
                freqs_gt[i] = self.pops[i].get_genotype_freq(gt)
            genotype_freqs[l] = freqs_gt
        return(genotype_freqs)
    def get_mean_fitness(self):
        fitness_array = np.zeros(len(self.pops))
        for i in range(len(self.pops)):
            fitness_array[i] = self.pops[i].get_mean_fitness()
        return(fitness_array)
    def plot_gene_freqs(self, ax=None, legend=False, **kwargs):
        gene_freqs = self.get_gene_freqs()
        if ax is None:
            for i,label in zip(range(gene_freqs.shape[1]), ['tub', 'nam7', 'VIII']):
                plt.plot(gene_freqs[:,i], label=label, **kwargs)
            plt.xlabel('generations')
            plt.ylabel('allele frequency')
            if legend:
                plt.legend()
        else:
            for i,label in zip(range(gene_freqs.shape[1]), ['tub', 'nam7', 'VIII']):
                ax.plot(gene_freqs[:,i], label=label, **kwargs)
            ax.set_xlabel('generations')
            ax.set_ylabel('allele frequency')
            if legend:
                ax.legend()
    def plot_genotype_freqs(self, ax=None, legend=False, **kwargs):
        for key,val in self.get_genotype_freqs().items():
            if ax is None:
                plt.plot(val, label=key, **kwargs)
                plt.xlabel('generations')
                plt.ylabel('genotype frequency')
                if legend:
                    plt.legend(title='genotype\n(tub/nmd/VIII)')
            else:
                ax.plot(val, label=key, **kwargs)
                ax.set_xlabel('generations')
                ax.set_ylabel('genotype frequency')
                if legend:
                    ax.legend(title='genotype\n(tub/nmd/VIII)')                
    def plot_mean_fitness(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.get_mean_fitness(), **kwargs)
            plt.xlabel('generations')
            plt.ylabel('mean fitness')            
        else:
            ax.plot(self.get_mean_fitness(), **kwargs)
            ax.set_xlabel('generations')
            ax.set_ylabel('mean fitness')            
