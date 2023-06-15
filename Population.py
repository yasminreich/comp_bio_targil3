import Individual
import Mutation
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from collections import deque
import multiprocessing


class Population:
    def __init__(self, train_data, train_labels, size, F1Threshold):
        self.size = size
        self.population = []
        self.numOfLayers = 6
        self.layersSizes = [16, 32, 128, 32, 16, 2]
        self.train_data = train_data
        self.train_labels = train_labels
        self.F1Threshold = F1Threshold

        self.__createInitialPop()

    def __createInitialPop(self):
        inputSize = self.train_data.shape[1]
        for i in range(self.size):
            nn = Individual.NN(inputSize, self.F1Threshold, self.numOfLayers,
                               self.layersSizes, self.train_data, self.train_labels)
            self.population.append(nn)

    # def train_and_calculate_fitness(self, p):
    #     p.train()
    #     p.calculateFitness()

    def runTrain(self):
        pool = multiprocessing.Pool()
        pool.map(train_and_calculate_fitness, self.population)
        # pool.close()
        pool.join()
        pool.close()

        # for p in self.population:
        #     p.train()
        #     p.calculateFitness()

    def dispachBestPeople(self, bestPeople):

        newPop = []
        for person in bestPeople:
            # create new copies of the best individual (5% of the new population)
            percent = person.fitness/10*100
            amount = math.ceil((self.size/100)*percent)
            if amount == 0:
                amount = 1
            for i in range(amount):
                newPop.append(person.deepcopy())

        return newPop

    # remove individuals with low fitness
    def __naturalSelection(self, percent):

        # devidor = self.__getDevidor(percentileNum)
        temp = []
        people_to_remove = int(self.size * percent)
        temp = self.population[:self.size - people_to_remove]
        self.population = temp

    def nextGen(self, deathThreshold, mutationChance):
        self.runTrain()
        self.population = sorted(
            self.population, key=lambda p: p.fitness, reverse=True)

        self.bestPerson = self.population[0]
        self.__naturalSelection(deathThreshold)

        newPopulation = self.dispachBestPeople(self.population[:5])

        popForCros = []
        # create vector of people for the crossover
        for person in self.population:
            fit = person.fitness*100
            for _ in range(int(fit)):
                popForCros.append(person.deepcopy())

        while True:
            # print("in crossover in next gen")
            print(len(newPopulation))
            parent1, parent2 = random.sample(popForCros, 2)
            child1, child2 = Mutation.crossover(parent1, parent2)
            newPopulation.append(child1)
            if len(newPopulation) >= self.size:
                break
            newPopulation.append(child2)
            if len(newPopulation) >= self.size:
                break

        self.population = newPopulation

        for person in self.population:
            if random.random() < mutationChance:
                Mutation.mutate_individual(person)


# Define a top-level function outside the class
def train_and_calculate_fitness(p):
    p.train()
    p.calculateFitness()


def read_data(file_path):
    samples = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('   ')
            if len(data) >= 2:
                sample = [int(bit) for bit in data[0].strip() if bit.isdigit()]
                label = int(data[1])

                # samples.append(np.array(sample).reshape(1, len(sample)))
                samples.append(np.array(sample))
                labels.append(label)

    return np.array(samples), np.array(labels)


def main():

    data, labels = read_data("nn0.txt")

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    popy = Population(train_data, train_labels, size=60, F1Threshold=0.3)
    deathThreshold = 0.4
    mutationChance = 0.8
    convergenceMax = 10
    convergenceCount = 0
    generationCounter = 0
    fitQueue = deque(maxlen=convergenceMax)
    epsilon = 0.001
    maxGen = 200
    while generationCounter < maxGen:
        generationCounter += 1

        popy.nextGen(deathThreshold=deathThreshold,
                     mutationChance=mutationChance)
        fitQueue.append(popy.bestPerson.fitness)
        print("best person fitness:", float(popy.bestPerson.fitness))
        print("best person accuracy:", float(popy.bestPerson.accuracy))
        if len(fitQueue) == convergenceMax:
            if popy.bestPerson.fitness - fitQueue[0] < epsilon:
                break

        # if popy.bestPerson.fitness == lastBestFit:
        #     convergenceCount += 1
        # else:
        #     convergenceCount = 0
        # lastBestFit = popy.bestPerson.fitness

    print(popy.bestPerson.test(test_data, test_labels))


if __name__ == "__main__":
    main()
