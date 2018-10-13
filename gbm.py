import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets.samples_generator import make_blobs

class GBR:

    def __init__(self, loss):

        self.loss = loss




# test the performance
def main():
    
    # create synthetic data
    N = 100
    X, y = make_blobs(n_samples=N, centers=2, n_features=2)
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))

    # visualise the data
    fig, ax = plt.subplots(1)
    grouped = df.groupby('label')
    for key, group in grouped:
        ax.scatter(group['x'], group['y'])
    plt.show()


if __name__ == '__main__':
    main()
