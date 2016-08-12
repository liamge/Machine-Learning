import numpy as np

class PCA:
    def fit(self,X):
        self.X = X
        self.m, self.n = X.shape[0], X.shape[1]
        Sigma = (1/self.m) * np.dot(X.T,X)
        self.U, self.S, V = np.linalg.svd(Sigma)

    def project(self,K):
        U_reduce = self.U[:,0:K]
        Z = np.dot(self.X, U_reduce)
        return Z

    def recover(self,Z,K):
        U_reduce = self.U[:,0:K]
        X_rec = np.dot(Z, U_reduce.T)
        return X_rec

    def variance(self,K):
        numerator = (1/self.m)*sum(self.S[0:K])
        denomenator = (1/self.m)*sum(self.S)
        return 1-(numerator/denomenator)

    def compute_K(self,min_var_loss):
        K = 1
        while self.variance(K) >= min_var_loss:
            K += 1
        return K