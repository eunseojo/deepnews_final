import numpy as np

'''
reads in glove input and saves a numpy array
'''

glove_path = "/Users/eunseo/Downloads/glove.6B/glove.6B.100d.txt"
matrix_save_path = "100d_glove_matrix"


f = open(glove_path)
glove_elements = f.read().split()

A = np.zeros(len(glove_elements) - 400000)
j = 0
for e, i in enumerate(glove_elements):

    if e % 101 == 0:
        continue
    else:
        A[j] = i
        j += 1

A_reshaped = np.reshape(A, (-1,100))

np.save(matrix_save_path, A_reshaped)

