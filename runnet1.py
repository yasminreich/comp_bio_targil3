import sys, pickle
import numpy as np


def read_data(file_path):
    samples = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('   ')
            if len(data) >= 2:
                sample = [int(bit) for bit in data[0].strip() if bit.isdigit()]

                # samples.append(np.array(sample).reshape(1, len(sample)))
                samples.append(np.array(sample))

    return np.array(samples)

@staticmethod
def load(filename):
    with open(filename, "rb") as file:
        nn = pickle.load(file)
    return nn

def main(wnet_path, data_path):
    nn = load(wnet_path)
    samples = read_data(data_path)

    predictions = nn.test(samples)
    with open("predictions1.txt", "w") as f:
        f.write('\n'.join(str(x) for x in list(predictions)))


if __name__ == "__main__":
    
    main(sys.argv[1], sys.argv[2])