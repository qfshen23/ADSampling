import numpy as np
import matplotlib.pyplot as plt
import os
from ivf import read_fvecs, source

# Use same dataset path structure as ivf.py
dataset = 'word2vec'  # Using deep1M as it's likely word vectors
path = os.path.join(source, dataset)
data_path = os.path.join(path, f'{dataset}_base.fvecs')

# Read the vectors
vectors = read_fvecs(data_path)

# Get first vector and plot it
first_vector = vectors[0]

# Calculate and print the L2 norm (length) of the first vector
vector_length = np.sqrt(np.sum(first_vector * first_vector))
print(f"L2 norm of first vector: {vector_length}")



