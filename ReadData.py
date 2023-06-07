import numpy as np

def read_data(file_path):
    samples = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('   ')
            if len(data) >= 2:
                sample = [int(bit) for bit in data[0].strip() if bit.isdigit()]
                label = int(data[1])
                
                samples.append(sample)
                labels.append(label)

    return np.array(samples), np.array(labels)

# if __name__ == '__main__':
#     file_path = 'nn0.txt'  # Replace with the actual path to your classified samples file
    
#     samples, labels = read_data(file_path)
    
#     print("First 20 Samples:")
#     print(samples[:20])
#     print("First 20 Labels:")
#     print(labels[:20])

