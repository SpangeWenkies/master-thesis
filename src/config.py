import numpy as np

# Magic numbers
sPlot = 'line'
iSeed = 12085278

# GARCH params
n = 10000
R = 1000
P = 1000
H = 10
reps = 1000
grid_size = 10000
dist = 't'
nu = np.array([5])
omega = np.array([0.01])
alpha = np.array([0.1])  # np.array([0.1]) #np.array([0.05, 0.1])
beta = np.array([0.6])

# bb7 params
theta = np.array([2])
delta = np.array([2])