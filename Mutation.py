import numpy as np
import Individual
import random
import ReadData

def mutate_individual(individual):
    random_number = random.randint(0, len(individual.weights)-1)
    shape = individual.weights[random_number].shape
    random_indices = tuple(np.random.randint(0, dim) for dim in shape)
    # print("the mutation site: [{}][{}]".format(random_number, random_indices))
    individual.weights[random_number][random_indices] += np.random.randn()
            
    return individual

def help_crossover_arrays(array1, array2, random_indices):
    # Create copies of the input arrays to avoid modifying them directly
    array1_crossover = array1.copy()
    array2_crossover = array2.copy()

    # Perform crossover from the random indices till the end of the arrays
    array1_crossover[random_indices[0]][random_indices[1]:] = array2[random_indices[0]][random_indices[1]:]
    array2_crossover[random_indices[0]][random_indices[1]:] = array1[random_indices[0]][random_indices[1]:]
    array1_crossover[(random_indices[0]+1):] = array2[(random_indices[0]+1):]
    array2_crossover[(random_indices[0]+1):] = array1[(random_indices[0]+1):]

    return array1_crossover, array2_crossover

def crossover(individual1, individual2):
    # Create copies of the parents to preserve their original values
    child1 = individual1.deepcopy()
    child2 = individual2.deepcopy()
    #choose a random cut-off
    random_number = random.randint(0, len(individual1.weights)-1)
    shape = individual1.weights[random_number].shape
    random_indices = tuple(np.random.randint(0, dim) for dim in shape)
    print("the cut-off site: [{}][{}]".format(random_number, random_indices))
    weights1 = []
    weights2 = []
    if random_number > 0:
        for i in range(random_number):
            weights1.append(individual1.weights[i])
            weights2.append(individual2.weights[i])
            print(":)")
    crossed_arr1, crossed_arr2 = help_crossover_arrays(individual1.weights[random_number], individual2.weights[random_number], random_indices)
    weights1.append(crossed_arr1)
    weights2.append(crossed_arr2)
    if random_number < len(individual1.weights)-1:
        for j in range(random_number+1, len(individual1.weights)):
            weights1.append(individual2.weights[j])
            weights2.append(individual1.weights[j])
            print(":(")
    child1.weights, child2.weights = weights1, weights2
    return child1, child2

if __name__ == '__main__':
    train_data, train_labels = ReadData.read_data("nn0.txt")
    indi1 = Individual.NN(16, 2, [2, 3], train_data, train_labels)
    indi2 = Individual.NN(16, 2, [2, 3], train_data, train_labels)
    print("indi1 weights: ")
    print(indi1.weights)
    print("indi2 weights: ")
    print(indi2.weights)
    child1, child2 = crossover(indi1, indi2)
    print("child1 weights: ")
    print(child1.weights)
    print("child2 weights: ")
    print(child2.weights)