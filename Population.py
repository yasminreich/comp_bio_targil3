import Individual
import Mutation
import numpy as np
import math
import random, json
from sklearn.model_selection import train_test_split
from collections import deque
import multiprocessing


class Population:
    def __init__(self, train_data, train_labels, size, F1Threshold):
        self.size = size
        self.population = []
        self.numOfLayers = 4
        self.layersSizes = [6, 8, 8, 2]
        self.numOfLayers = 2
        self.layersSizes = [8, 2]

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


        for p in self.population:
            p.train()
            p.calculateFitness()

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
        self.population = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        self.__naturalSelection(deathThreshold)


        # lamark
        for person in self.population:
            newPerson = person.deepcopy()
            for i in range(3):
                Mutation.mutate_individual(newPerson)
            newPerson.train()

            if newPerson.fitness > person.fitness:
                person.fitness = newPerson.deepcopy()

        self.bestPerson = self.population[0]
        newPopulation = self.dispachBestPeople(self.population[:5])

        popForCros = []
        # create vector of people for the crossover
        for person in self.population:
            fit = person.fitness*100
            for _ in range(int(fit)):
                popForCros.append(person.deepcopy())

        while True:
            # print("in crossover in next gen")
            # print(len(newPopulation))
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

# function to add to JSON
def write_json(new_data, filename='results.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["emp_details"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def main():



    data, labels = read_data("nn0.txt")

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    popSize = 200

    popy = Population(train_data, train_labels, size=popSize, F1Threshold=0.3)
    deathThreshold = 0.6
    mutationChance = 0.8
    convergenceMax = 10
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
            if popy.bestPerson.accuracy - fitQueue[0] < epsilon:
                break

        # if popy.bestPerson.fitness == lastBestFit:
        #     convergenceCount += 1
        # else:
        #     convergenceCount = 0
        # lastBestFit = popy.bestPerson.fitness


    print(popy.bestPerson.test(test_data, test_labels))

    result = {"deathThreshold":deathThreshold,  "mutationChance":mutationChance,\
         "layers": popy.layersSizes, "train accuracy": popy.bestPerson.accuracy,\
             "test accuracy": popy.bestPerson.test(test_data, test_labels), \
                 "num of gen": generationCounter, "epsilon":0.001, "max gen": maxGen}

    write_json(result)

if __name__ == "__main__":
    main()
