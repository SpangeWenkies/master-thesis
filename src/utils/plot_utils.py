import matplotlib.pyplot as plt

###########################################################
# Plot

def Plot(vX, sTitle, bLine):
    """
    Purpose:
        Plot series vX

    Inputs:
        vX              time series
        bLine           boolean: lineplot (1) or histogram (0)


    Output:
        Plot of realisations process

    """

    lFigureSettings = {'figsize': (10, 6), 'dpi': 70, 'titlefontsize': 16, 'axisfontsize': 14}
    iBins = 30  # int(np.floor(len(vX)/25))

    plt.figure(figsize=lFigureSettings['figsize'], dpi=lFigureSettings['dpi'])
    if (bLine == True):
        plt.plot(vX, c='blue')
        plt.xlabel('$t$', fontsize=lFigureSettings['axisfontsize'])
        plt.ylabel('$y_t$', fontsize=lFigureSettings['axisfontsize'])
    else:
        plt.hist(vX, iBins)
    plt.title(sTitle, fontsize=lFigureSettings['titlefontsize'])
    plt.show()