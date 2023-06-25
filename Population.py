import Individual
import Mutation
import math
import random

#the neural netwok set that the genetic algorithen is optimizing
class Population:
    def __init__(self, train_data, train_labels, size, layersSizes):
        self.size = size
        self.population = []
        
        self.layersSizes = layersSizes
        self.numOfLayers = len(self.layersSizes)

        self.train_data = train_data
        self.train_labels = train_labels

        self.__createInitialPop()
    #create initial population in the given size using the initializing of the individual
    def __createInitialPop(self):
        inputSize = self.train_data.shape[1]
        for i in range(self.size):
            nn = Individual.NN(inputSize, self.numOfLayers,
                               self.layersSizes, self.train_data, self.train_labels)
            self.population.append(nn)

    #run train on all the individuals in the population
    def runTrain(self):

        for p in self.population:
            p.train()

    #dispatch the best 5 individuals in the population to the next generation (elitisem)
    def dispachBestPeople(self, bestPeople):
        newPop = []
        for person in bestPeople:
            # create new copies of the best individual (5% of the new population)
            percent = person.accuracy/10*100
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

    #get the next generaion population
    def nextGen(self, deathThreshold, mutationChance):
        #evaluate the population
        self.runTrain()
        self.population = sorted(self.population, key=lambda p: p.accuracy, reverse=True)
        self.__naturalSelection(deathThreshold)

        # lamarkian genetic algorithem
        for person in self.population:
            newPerson = person.deepcopy()
            for i in range(3):
                Mutation.mutate_individual(newPerson)
            newPerson.train()
            if newPerson.accuracy > person.accuracy:
                person.person = newPerson.deepcopy()
        
        self.bestPerson = self.population[0]
        newPopulation = self.dispachBestPeople(self.population[:5])

        popForCros = []
        # create vector of people for the crossover
        for person in self.population:
            fit = person.accuracy*100
            for _ in range(int(fit)):
                popForCros.append(person.deepcopy())
        #do cross-over to fill the rest of the next generation population
        while True:
            parent1, parent2 = random.sample(popForCros, 2)
            child1, child2 = Mutation.crossover(parent1, parent2)
            newPopulation.append(child1)
            if len(newPopulation) >= self.size:
                break
            newPopulation.append(child2)
            if len(newPopulation) >= self.size:
                break

        self.population = newPopulation
        #mutate according to the mutation rate
        for person in self.population:
            if random.random() < mutationChance:
                Mutation.mutate_individual(person)