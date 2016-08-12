import numpy as np
from random import randint

class KMeans:
    def fit(self,X):
        # Initiates some useful variables
        self.X = X
        self.m, self.n = X.shape[0], X.shape[1]

    def init_centroids(self,K):
        centroids = np.zeros((K,self.n))
        for i in range(K):
            centroids[i] = self.X[randint(0,self.m)]
        return centroids

    def squared_distance(self,x1,x2):
        return sum((x1-x2)**2)

    def find_centroids(self,centroids):
        K = centroids.shape[0]
        idx = np.zeros((self.m,1))
        for i in range(self.m):
            min_dist = float("inf")
            for k in range(K):
                distance = self.squared_distance(self.X[i],centroids[k])
                if distance < min_dist:
                    idx[i] = k
                    min_dist = distance
        return idx

    def compute_centroids(self,idx,K):
        centroids = np.zeros((K,self.n))
        for k in range(K):
            indices = idx == k
            centroids[k] = sum(self.X*indices) / sum(indices)
        return centroids

    def main(self,max_iters,K):
        init_centroids = self.init_centroids(K)
        for i in range(max_iters):
            print("K-Means iteration: %d" % (i+1))
            idx = self.find_centroids(init_centroids)
            centroids = self.compute_centroids(idx, K)
        return centroids, idx

    def graph(self,feature1,feature2,idx):
        # Plots two dimensional scatter
        import matplotlib.pyplot as plt
        plt.scatter(feature1,feature2,c=idx)