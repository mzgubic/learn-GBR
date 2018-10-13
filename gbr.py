import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets.samples_generator import make_blobs

class LossFunction:
    
    def __init__(self):
        pass

class MeanSquareLoss(LossFunction):

    def __call__(self, y, y_pred):

        return 0.5 * np.mean((y - ypred)**2)

    def negative_gradient(self, y, y_pred):
        return y - y_pred

class GBR:

    def __init__(self, loss, learning_rate=0.1, n_estimators=2, criterion='friedman_mse', random_state='42'):

        if loss == 'ls':
            self.loss = MeanSquareLoss()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.estimators = [] # estimators
        self.gammas = [] # and their weights
        self.criterion = criterion

    def fit(self, X, y):

        # fit the initial tree
        self.estimators.append(DecisionTreeRegressor(criterion=self.criterion, random_state=self.random_state))
        self.gammas.append(1.0)
        self.estimators[0].fit(X, y)

        # fit the rest of them
        for i in range(1, self.n_estimators):
            print('Fitting ', i, 'th estimator in the sequence')
            tree_i = DecisionTreeRegressor(criterion=self.criterion, random_state=self.random_state)

            # TODO: change below to make it actually work
            tree_i.fit(X, y)
            self.estimators.append(tree_i)
            self.gammas.append(1.0)


    def predict(self, X, n=None):

        # predict on all if it is not specified
        if n==None:
            n = len(self.estimators)

        # sum predictions over all classifiers up to n
        predictions = np.zeros(shape=X.shape[0])
        for i, est in enumerate(self.estimators):
            print('Predicting estimator {i}/{n}.'.format(i=i+1, n=n))
            preds = est.predict(X)
            predictions += self.gammas[i] * preds
            print('Current predictions', predictions[:5])

        return predictions


def plot_decision_surface(clf, X, y, plot_step = 0.2, cmap='coolwarm', figsize=(12,8)):
    """Plot the prediction of clf on X and y, visualize training points"""
    print("##########################")
    print("Plotting decision boundary")
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
    which = 'gbr'
    if which == 'dtr':
        clf = DecisionTreeRegressor(criterion='friedman_mse', random_state=random_state)
    elif which == 'gbr':
        clf = GBR(loss='dummy', random_state=random_state)
    clf.fit(X, y)

    # visualise the data
    plot_decision_surface(clf, X, y)


if __name__ == '__main__':
    main()
