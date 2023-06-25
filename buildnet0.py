from Population import Population
import numpy as np
import json, os, sys
from sklearn.model_selection import train_test_split
from collections import deque

#read the input file and devide to samples and labels in np arrays
def read_data(file_path):
    samples = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('   ')
            if len(data) >= 2:
                sample = [int(bit) for bit in data[0].strip() if bit.isdigit()]
                label = int(data[1])
                samples.append(np.array(sample))
                labels.append(label)

    return np.array(samples), np.array(labels)

#run the the genetic algorithen to find the weights that gives the best accurecy on the test set
def main(train_path, test_path):

    train_data, train_labels = read_data(train_path)
    test_data, test_labels = read_data(test_path)
    #the hiper parameters we found that worked the best
    popSize = 200
    layersSizes=[4,2]
    deathThreshold = 0.8
    mutationChance = 0.8
    convergenceMax = 10
    epsilon = 0.0001
    maxGen = 300

    while True:
        popy = Population(train_data, train_labels, size=popSize, layersSizes=layersSizes)
        generationCounter = 0
        fitQueue = deque(maxlen=convergenceMax)
        #keep searching until converges or reaching the max gens
        while generationCounter < maxGen:
            generationCounter += 1

            popy.nextGen(deathThreshold=deathThreshold,
                        mutationChance=mutationChance)
            fitQueue.append(popy.bestPerson.accuracy)
            print("best person accuracy:", float(popy.bestPerson.accuracy))
            #check if converges
            if len(fitQueue) == convergenceMax:
                if abs(popy.bestPerson.accuracy - fitQueue[0]) <= epsilon:
                    break
        

        test_accuracy = np.mean(popy.bestPerson.test(test_data) == test_labels)
        if test_accuracy > 0.98:
            break
    #save the best neural network found
    popy.bestPerson.save("wnet0.txt")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])