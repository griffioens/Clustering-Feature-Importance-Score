import numpy as np
from sklearn.base import MultiOutputMixin, BaseEstimator
import skfuzzy as fuzz

class FuzzyCMeans(MultiOutputMixin, BaseEstimator):

    def __init__(self, n_centers=3, seed=0):
        
        self.n_centers = n_centers
        self.centroids = None
        self.seed = seed
  
    def fit(self, X_train, Y_train=None):
        
        self.centroids, _, _, _, _, _, _ = fuzz.cluster.cmeans(X_train.T, self.n_centers, m=2, error=0.005, maxiter=1000, seed=self.seed)
        
        return self
    
    def predict(self, X):

        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T, self.centroids, m=2, error=0.005, maxiter=1000, seed=self.seed)

        return u.T
    
    def f_importance(self, X):

        f_values = np.zeros(X.shape[1])
        for xi in enumerate(self.centroids):
            for xj in enumerate(self.centroids):
                f_values += np.abs(xi[1]-xj[1])
        
        return f_values/np.sum(f_values)