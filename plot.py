import os
import re
import json
import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np

def savePlot(plt, plotParam, timeSpent):
    # Set the path to your directory containing the .png files
    path = "./plots"

    # Find the largest integer in the existing filenames
    existing_ints = []
    for filename in os.listdir(path):
        match = re.match(r"#(\d+)\.png", filename)
        if match:
           existing_ints.append(int(match.group(1)))
    if existing_ints:
        next_int = max(existing_ints) + 1
    else:
        next_int = 1

    title = plotParam['file'] + ' Network ' + f"#{next_int}" + "  \nTime it took: " + str(int(timeSpent)) + " seconds" + "\nFitness function: " + plotParam['fitness_func']
    plt.title(title)
    plt.savefig(os.path.join(path, f"#{next_int}.png"))

def printAndSavePlot(plotParam, timeSpent):
    best, = plt.plot(plotParam['generations'], plotParam['allBestFitnesses'], 'g-', label = 'best')
    mean, = plt.plot(plotParam['generations'], plotParam['allAvgFitnesses'], 'b-', label = 'mean')
    worst, = plt.plot(plotParam['generations'], plotParam['allWorstFitnesses'], 'r-', label = 'worst')

    plt.legend([best, mean, worst], ['Best', 'Mean', 'Worst'])

    plt.xlabel("Number of generations")
    plt.ylabel(plotParam['fitness_func'].capitalize() + " value")
    savePlot(plt, plotParam, timeSpent)

    plt.show()