from random import randint

from MyChromosome import MyChromosome


class GA:
    def __init__(self, fitness_func, param=None):
        self.__param = param
        self.__fitness_function = fitness_func
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self, network):
        for _ in range(0, self.__param['popSize']):
            c = MyChromosome()
            c.init_representation(network)
            self.__population.append(c)

    def evaluation(self, network):
        for crom in self.__population:
            crom.fitness = self.__fitness_function(crom.representation, network)

    def best_chromosome(self):
        bestc = self.__population[0]
        for crom in self.__population:
            if crom.fitness > bestc.fitness:
                bestc = crom
        return bestc
    
    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if (c.fitness < best.fitness):
                best = c
        return best
    
    def averageFitness(self):
        avg = 0.0
        for c in self.__population:
            avg += c.fitness
        return avg/self.__param['popSize']

    def selection(self):
        pos_1 = randint(0, self.__param['popSize'] - 1)
        pos_2 = randint(0, self.__param['popSize'] - 1)
        if self.__population[pos_1].fitness > self.__population[pos_2].fitness:
            return pos_1
        else:
            return pos_2

    def one_generation(self, network):
        new_population = []
        for _ in range(self.__param['popSize']):
            crom_1 = self.__population[self.selection()]
            crom_2 = self.__population[self.selection()]
            off = crom_1.crossover(crom_2)
            off.mutation(self.__param['mutFactor'])
            new_population.append(off)
        self.__population = new_population
        self.evaluation(network)