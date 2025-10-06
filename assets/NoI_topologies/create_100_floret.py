# --------------------------------------------
# Generate 100x100 Adjacency Matrix from Path
# --------------------------------------------
# This script constructs an undirected adjacency matrix for a 10x10 network (100 nodes total)
# based on a predefined path. It outputs the matrix in space-separated format suitable for use
# in simulators or custom parsers expecting whitespace-delimited input.
# --------------------------------------------

import numpy as np

# Define the path
path = [0, 1, 11, 21, 31, 41, 42, 32, 33, 43, 44, 34, 24, 23, 22, 12, 2, 3, 13, 14, 4, 5, 15, 16, 6, 7, 17, 27, 26, 25, 35, 
        45, 46, 36, 37, 47, 48, 38, 28, 18, 8, 9, 19, 29, 39, 49, 59, 58, 57, 56, 55, 65, 66, 67, 68, 69, 79, 78, 77, 76, 86, 
        87, 88, 89, 99, 98, 97, 96, 95, 85, 75, 74, 84, 94, 93, 92, 91, 90, 80, 81, 82, 83, 73, 72, 71, 70, 60, 61, 62, 63, 64, 54, 53, 
        52, 51, 50, 40, 30, 20, 10, 0]

# Initialize 100x100 matrix
adj_matrix = np.zeros((100, 100), dtype=int)

# Fill in connections
for i in range(len(path) - 1):
    u, v = path[i], path[i + 1]
    adj_matrix[u, v] = 1
    adj_matrix[v, u] = 1  # undirected graph

# Save to file without commas (space-separated)
with open("adj_matrix_100x100.txt", "w") as f:
    for row in adj_matrix:
        f.write(" ".join(map(str, row)) + "\n")
