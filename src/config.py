import numpy as np

# Magic numbers
sPlot = 'line'
iSeed = 12085278

# GARCH params
n = 10000
dist = 't'
nu = np.array([5])
omega = np.array([0.01])
alpha = np.array([0.1])  # np.array([0.1]) #np.array([0.05, 0.1])
beta = np.array([0.6])

# bb7 params
theta = np.array([4])
delta = np.array([6])