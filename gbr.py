import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets.samples_generator import make_blobs

class GBR:

    def __init__(self, loss, n_estimators=2, criterion='friedman_mse', random_state='42'):

        self.loss = loss
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []
        self.criterion = criterion

    def fit(self, X, y):

        # fit the initial tree
        self._estimators.append(DecisionTreeRegressor(criterion=self.criterion))
        self._estimators.fit(X, y)

        # fit the rest of them

    def predict(self, X):

        predictions = np.zeros(shape=y.shape)
        for est in self.estimators:
            preds = est.predict(X)
            predictions += preds

        return predictions


def plot_decision_surface(clf, X, y, plot_step = 0.2, cmap='coolwarm', figsize=(12,8)):
    """Plot the prediction of clf on X and y, visualize training points"""
    plt.figure(figsize=figsize)
    x0_grid, x1_grid = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, plot_step),
                         np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, plot_step))
    y_pred_grid = clf.predict(np.stack([x0_grid.ravel(), x1_grid.ravel()],axis=1)).reshape(x1_grid.shape)
    plt.contourf(x0_grid, x1_grid, y_pred_grid, cmap=cmap, alpha=0.5)  
    y_pred = clf.predict(X)    
    plt.scatter(*X.T, c=y, cmap=cmap, marker='x') 
    plt.show()

def main():
    
    # create synthetic data
    N_p = 1000
    N_c = 3
    random_state=42
    np.random.seed(random_state)
    X, y = make_blobs(n_samples=N_p, centers=N_c, n_features=2, center_box=(-5, 5), random_state=random_state)
    y = y + np.random.normal(0, 0.1, y.shape[0])
    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))

    # train a model
    clf = DecisionTreeRegressor(random_state=random_state)
    clf.fit(X, y)

    # visualise the data
    plot_decision_surface(clf, X, y)


if __name__ == '__main__':
    main()
