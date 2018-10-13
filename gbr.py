import numpy as np
import pandas as pd
import scipy.optimize as spo
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

class LossFunction:
    
    def __init__(self):
        pass

class MeanSquareLoss(LossFunction):

    def __call__(self, y, y_pred):
        return 0.5 * np.mean((y - y_pred)**2)

    def negative_gradient(self, y, y_pred):
        return y - y_pred

class GBR:

    def __init__(self, loss, learning_rate=0.1, n_estimators=2, criterion='friedman_mse', random_state='42', max_depth=3):

        if loss == 'ls':
            self.loss = MeanSquareLoss()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators = [] # estimators
        self.gammas = [] # and their weights
        self.criterion = criterion

    def fit(self, X, y):

        # fit the initial tree
        #print('Fitting 1 st estimator in the sequence')
        self.estimators.append(DecisionTreeRegressor(criterion=self.criterion, random_state=self.random_state,
                                                     max_depth=self.max_depth))
        self.gammas.append(1.0)
        self.estimators[0].fit(X, y)
        pred_y_vals = self.estimators[0].predict(X)

        # fit the rest of them
        for i in range(2, self.n_estimators):
            #print()
            #print('Fitting', i, 'th estimator in the sequence')
            tree_i = DecisionTreeRegressor(criterion=self.criterion, random_state=self.random_state,
                                           max_depth=self.max_depth)

            # fit the tree
            y_prev_pred = self.predict(X, n=i-1)
            y_i = self.loss.negative_gradient(y, y_prev_pred)
            tree_i.fit(X, y_i)

            # fit the normalisation factor
            y_pred_i = tree_i.predict(X)
            def loss(gamma):
                return self.loss(y, y_prev_pred + gamma*y_pred_i)
            best_gamma = spo.minimize(loss, 1.0).x[0]

            # append the fitted values (estimators and gamma factors)
            self.estimators.append(tree_i)
            self.gammas.append(best_gamma*self.learning_rate)

    def predict(self, X, n=None):

        # predict on all if it is not specified
        if n==None:
            n = len(self.estimators)
        #print('    Called to predict {n}.'.format(n=n))

        # sum predictions over all classifiers up to n
        predictions = np.zeros(shape=X.shape[0])
        for i, est in enumerate(self.estimators):
            if i == n:
                break
            #print('    Predicting estimator {i}/{n}.'.format(i=i+1, n=n))
            preds = est.predict(X)
            predictions += self.gammas[i] * preds
            #print('    Current predictions', predictions[:5])

        return predictions


def plot_decision_surface(clf, X, y, plot_step = 0.05, cmap='coolwarm', figsize=(12,8)):
    """Plot the prediction of clf on X and y, visualize training points"""
    print("##########################")
    print("Plotting decision boundary")
    vmin = np.min(y)
    vmax = np.max(y)
    plt.figure(figsize=figsize)
    x0_grid, x1_grid = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, plot_step),
                         np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, plot_step))
    y_pred_grid = clf.predict(np.stack([x0_grid.ravel(), x1_grid.ravel()],axis=1)).reshape(x1_grid.shape)
    plt.contourf(x0_grid, x1_grid, y_pred_grid, cmap=cmap, vmin=vmin, vmax=vmax)
    y_pred = clf.predict(X)    
    plt.scatter(*X.T, c=y, cmap=cmap, marker='x', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

def main():
    
    # create synthetic data
    random_state = 42
    np.random.seed(random_state)
    N = 1000
    N_groups = 3
    X, y = make_blobs(n_samples=N, centers=N_groups, n_features=2, center_box=(-5, 5), random_state=random_state)
    y = y + np.random.normal(0, 0.1, y.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # model parameters
    max_depth = 3
    n_estimators = 100
    criterion='friedman_mse'
    learning_rate = 0.1

    # choose the model
    kwargs = {'loss':'ls', 'random_state':random_state, 'n_estimators':n_estimators, 'max_depth':max_depth,
              'criterion':criterion, 'learning_rate':learning_rate}
    home_cooked_gbr = GBR(**kwargs)
    sklearn_gbr = GradientBoostingRegressor(**kwargs)

    # train the models
    home_cooked_gbr.fit(X_train, y_train)
    sklearn_gbr.fit(X_train, y_train)

    # visualise the data
    plot_decision_surface(home_cooked_gbr, X_train, y_train)
    plot_decision_surface(sklearn_gbr, X_train, y_train)

    # see how the model does on the test set
    home_pred_y = home_cooked_gbr.predict(X_test)
    sklearn_pred_y = sklearn_gbr.predict(X_test)
    print('Test loss for home cooked:', home_cooked_gbr.loss(y_test, home_pred_y))
    print('Test loss for sklearn    :', home_cooked_gbr.loss(y_test, sklearn_pred_y))



if __name__ == '__main__':
    main()
