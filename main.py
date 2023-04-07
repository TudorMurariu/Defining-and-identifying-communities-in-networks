import os
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from plot import *
import time

from GeneticAlg import GA

warnings.simplefilter('ignore')


def get_file_path(name):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'data', name, name + ".gml")
    return file_path


def read_graph_from_file(file_name):
    file_path = get_file_path(file_name)
    graph = nx.read_gml(file_path, label='id')
    return graph


def plot_network(G, communities):
    np.random.seed(333)

    pos = nx.spring_layout(G)
    # nx.graphviz_layout
    # graph
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_size=800, cmap=plt.cm.RdYlBu, node_color=communities)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()


def modularity(communities, network):
    no_nodes = nx.number_of_nodes(network)
    mat = nx.to_numpy_array(network)
    degrees = nx.degree(network)
    no_edges = nx.number_of_edges(network)
    M = 2 * no_edges
    Q = 0.0
    for i in range(0, no_nodes):
        for j in range(0, no_nodes):
            if communities[i] == communities[j]:
                Q += (mat[i][j] - degrees[i] * degrees[j] / M)
    return Q * 1 / M


# Diferenta dintre modularitatea simpla si aceasta este faptul ca aceasta incearca sa gaseasca si comunitatile mici
# din comunitatile mari. Parametrul lmbd, care este default 0.5, ne spune cat de dens este graful:  0 - nu exista
# muchii, 1 - e graf complet.
def modularity_density(communities, network, lmbd=0.5, num_com=2):
    my_communities = [[] for _ in range(num_com)]
    for i in range(nx.number_of_nodes(network)):
        my_communities[(communities[i] - 1) % num_com].append(i)
    Q = 0.0
    for community in my_communities:
        sub = nx.subgraph(network, community)
        sub_n = sub.number_of_nodes()
        interior_degrees = []
        exterior_degrees = []
        for node in sub:
            interior_degrees.append(sub.degree(node))
            exterior_degrees.append(network.degree(node) - sub.degree(node))
        try:
            Q += (1 / sub_n) * ((2 * lmbd * np.sum(interior_degrees)) - (2 * (1 - lmbd) * np.sum(exterior_degrees)))
        except ZeroDivisionError:
            pass
    return Q


# Incearca, de asemenea, sa evite limita de rezolutie, gasind si comunitatile mici din comunitatile mari.
# Se bazeaza pe distributia Student - se calculeaza raritatea statistica a unei comunitati.
def z_modularity(communities, network, num_com=2):
    my_communities = [[] for _ in range(num_com)]
    for i in range(nx.number_of_nodes(network)):
        my_communities[(communities[i] - 1) % num_com].append(i)
    edges = network.number_of_edges()
    Q = 0.0
    mmc = 0
    dc2m = 0
    for community in my_communities:
        sub = nx.subgraph(network, community)
        sub_n = sub.number_of_nodes()
        dc = 0
        for node in sub:
            dc += network.degree(node)
        mmc = sub_n / edges
        dc2m += (dc / (2 * edges)) ** 2
    try:
        Q = (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))
    except ZeroDivisionError:
        pass
    return Q


def run_ga(network, ga_param, fitness_func, file):
    ga = GA(fitness_func, ga_param)
    ga.initialisation(network)
    print("Initialized all chromosomes")
    ga.evaluation(network)

    # Plotting params
    plotParam = {'file': file, 'fitness_func': fitness_func.__name__, 'allBestFitnesses' : [], 'allWorstFitnesses' : [], 'allAvgFitnesses' : [], 'generations': [], 'bestChromosome': []}

    plotParam['allBestFitnesses'].append(ga.best_chromosome().fitness)
    plotParam['allWorstFitnesses'].append(ga.worstChromosome().fitness)
    plotParam['allAvgFitnesses'].append(ga.averageFitness())
    plotParam['generations'].append(0)

    best_crom = ga.best_chromosome()

    for generation in range(ga_param['noGen']):
        ga.one_generation(network)

        current_best = ga.best_chromosome()
        plotParam['allBestFitnesses'].append(ga.best_chromosome().fitness)
        plotParam['allWorstFitnesses'].append(ga.worstChromosome().fitness)
        plotParam['allAvgFitnesses'].append(ga.averageFitness())
        plotParam['generations'].append(generation)
        # print(str(generation + 1) + ' Current best: ' + ' \nFitness: '
        #       + str(current_best.fitness))

        if current_best.fitness > best_crom.fitness:
            best_crom = current_best

    plotParam['bestChromosome'] = best_crom

    return plotParam


if __name__ == '__main__':
    crtDir = os.getcwd()

    # tests()
    # file = input("Nume fisier: ")
    file = 'lobster'

    network_ = read_graph_from_file(file)
    network_aux = network_.copy()

    ga_params = {'popSize': 50, 'noGen': 200, 'mutFactor': 30}

    stTime = time.time()
    plotParam = run_ga(network_aux, ga_params, z_modularity, file)
    timeSpent = time.time() - stTime
    print("--- TOTAL %s seconds ---" %(timeSpent))

    printAndSavePlot(plotParam, timeSpent)

    print(plotParam['bestChromosome'])
    print("Nr. comunitati: " + str(len(np.unique(plotParam['bestChromosome'].representation))))
    plot_network(network_, plotParam['bestChromosome'].representation)