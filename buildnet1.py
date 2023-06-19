from Population import Population
import numpy as np
import json, os, sys
from sklearn.model_selection import train_test_split
from collections import deque

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

# # function to add to JSON
# def write_json(new_data, filename='results.json'):

#     # Check if the file exists
#     if os.path.exists(filename):
#         # File exists, so load its contents
#         with open(filename, "r") as file:
#             results = json.load(file)
#     else:
#     # File doesn't exist, create an empty data structure
#         results = []

#     # Update the existing data structure
#     results.append(new_data)

#     # Save the updated data back to the file
#     with open(filename, "w") as file:
#         json.dump(results, file)


def main(train_path, test_path):

    train_data, train_labels = read_data(train_path)
    test_data, test_labels = read_data(test_path)

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

        while generationCounter < maxGen:
            generationCounter += 1

            popy.nextGen(deathThreshold=deathThreshold,
                        mutationChance=mutationChance)
            fitQueue.append(popy.bestPerson.accuracy)
            print("best person accuracy:", float(popy.bestPerson.accuracy))
            if len(fitQueue) == convergenceMax:
                if abs(popy.bestPerson.accuracy - fitQueue[0]) <= epsilon:
                    break
        

        test_accuracy = np.mean(popy.bestPerson.test(test_data) == test_labels)
        if test_accuracy > 0.98:
            break
        
    popy.bestPerson.save("wnet1.txt")

if __name__ == "__main__":

    main(sys.argv[1], sys.argv[2])