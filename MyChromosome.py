import random

import networkx as nx


class MyChromosome:
    def __init__(self):
        self.__fitness = 0.0
        self.__representation = []

    @property
    def representation(self):
        return self.__representation

    @property
    def fitness(self):
        return self.__fitness

    @representation.setter
    def representation(self, chromosome_rep):
        self.__representation = chromosome_rep

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def normalize_rep(self):
        unique_nums = list(dict.fromkeys(self.__representation))
        self.__representation = [unique_nums.index(value) + 1 for value in self.__representation]

    def crossover(self, other):
        offspring = MyChromosome()
        chosen_community = random.choice(self.__representation)
        offspring.representation = [source if source == chosen_community else destination
                                    for source, destination in zip(self.representation, other.representation)]
        offspring.normalize_rep()
        return offspring

    def mutation(self, mutation_rate):
        rnd_chance = random.randint(0, 100)
        if rnd_chance < mutation_rate:
            poz_1 = random.randint(0, len(self.__representation) - 1)
            poz_2 = random.randint(0, len(self.__representation) - 1)
            self.__representation[poz_1], self.__representation[poz_2] =\
                self.__representation[poz_2], self.__representation[poz_1]
            self.normalize_rep()

    def init_representation(self, network):
        size = nx.number_of_nodes(network)
        self.__representation = [0 for _ in range(size)]
        for i in sorted(nx.nodes(network)):
            c = random.randint(1, size)
            self.__representation[i] = c
            for j in sorted(network.neighbors(i)):
                self.__representation[j] = c

    def __str__(self):
        return '\nChromosome: ' + str(self.__representation) + ' --- fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__representation == c.__repres and self.__fitness == c.__fitness