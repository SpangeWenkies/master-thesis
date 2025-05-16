import numpy as np
import matplotlib.pyplot as plt

from GarchSimv2 import simulate_GARCH


def sim_study():

    # Cleaning
    plt.close('all')

    # Magic numbers
    iT = 3250
    iSeed = 12085278
    dOmega = np.array([0.1])
    vAlpha = np.array([0.1])
    vBeta = np.array([0.6])
    dNu = 6
    weightFuncEnd = 5 * np.pi   # higher means more dist switches and less gradual change in weights

    # Set seed
    np.random.seed(iSeed)

    # creating a sinusoid weight series
    x1 = np.linspace(0, weightFuncEnd, iT)
    vW = 0.5 * np.sin(x1) + 0.5

    # Sim some GARCH models
    vY1, vY1_var = simulate_GARCH(iT, dOmega, vAlpha, vBeta, "normal", dNu)
    vY2, vY2_var = simulate_GARCH(iT, dOmega, vAlpha, vBeta, "t", dNu)

    # Plotting the two series
    plt.plot(vY1)
    plt.show()
    plt.close()

    plt.plot(vY2)
    plt.show()
    plt.close()

    # Creating a regime switching series using the defined sinusoid weight function
    vY = np.zeros(iT)
    for t in range(0, iT):
        vY[t] = vW[t] * vY1[t] + (1 - vW[t]) * vY2[t]

    # Plotting the regime switching series and the regime
    plt.plot(vY, color="blue", label="y")
    plt.plot(vW, color="black", label="Weights")
    plt.title("Normal Garch(1,1) + Student-t Garch(1,1) (df=6) , N_w = 5pi")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == '__main__':

    sim_study()

