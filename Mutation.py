import numpy as np
from Individual import NN

def mutate_individual(individual, mutation_rate):
    for i in range(len(individual.weights)):
        # Check if mutation should occur based on the mutation rate
        if np.random.rand() < mutation_rate:
            # Mutate the weights by adding random values drawn from a standard normal distribution
            individual.weights[i] += np.random.randn(*individual.weights[i].shape)
            
    return individual

def crossover_same_size(individual1, individual2):
    # Create copies of the parents to preserve their original values
    child1 = NN(numOfLayers=0, layersSizes=[], activationFunctions=[], samples=[], lables=[])
    child2 = NN(numOfLayers=0, layersSizes=[], activationFunctions=[], samples=[], lables=[])
    
    # Perform crossover on the weights of the parents
    for i in range(len(individual1.weights)):
        # Determine the crossover point
        crossover_point = np.random.randint(0, individual1.weights[i].shape[0])

        # Perform crossover by swapping the weights from the crossover point onwards
        child1.weights.append(np.concatenate((individual1.weights[i][:crossover_point], individual2.weights[i][crossover_point:])))
        child2.weights.append(np.concatenate((individual2.weights[i][:crossover_point], individual1.weights[i][crossover_point:])))

    return child1, child2
