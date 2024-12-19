import numpy as np


# 19 Dec 2024
def generate_spiral_planar_dataset():
    # generate spiral-shaped data points
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # number of columns in X
    X = np.zeros((m,D)) # data matrix where each row is a single example, 2 columns rep the coordindates 
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1)) # a range of indices, specifying where to insert the new data points into the X array and Y array
        # N evenly spaced numbers between 0 and 3.12 for red set and 3.12 and 6.24 for blue set
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta generate evenly spaced numbers over a specified interval and includes both the start and stop values 
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # r is the radius, and t is the angle in radians.
        # stacking the r*np.sin(t) and r*np.cos(t) arrays into a single array, where each row represents an (x, y) coordinate pair
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)] #Stacks these coordinates into a 2D X array, to specific positions within the X array, concatenates arrays column-wise
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y