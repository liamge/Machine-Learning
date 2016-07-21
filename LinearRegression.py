import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class LinearRegression:
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.m = len(y)
        self.theta = 0
        # mean and std need to be same dimensions as X
        self.mean = np.zeros(self.X.size[0])
        self.std = np.zeros(self.X.size[0])

    def gradientDescent(self, alpha, num_of_iterations, featureNormalize = True, plot = True):
        Xtrans = self.X.transpose()
        cost_history = []
        if featureNormalize:
            X = self.featureNormalize(self.X)
        else:
            X = self.X
        for i in range(num_of_iterations):
            hypothesis = np.dot(X,self.theta)
            loss = hypothesis - y
            cost = np.sum((loss**2)/(2*self.m))
            cost_history.append(cost)
            print('Iteration #%d\tCost:%d' % (i, cost))
            gradient = np.dot(Xtrans, loss)/self.m
            self.theta = self.theta - alpha * gradient
        if plot:
            plt.figure(0)
            plt.plot(cost_history, 'r')
            plt.ylabel('Cost')
            plt.xlabel('Iterations')
            plt.title('Cost function with Gradient Descent')
            plt.show()
            plt.figure(1)
            plt.plot(self.X, self.y,'o')
            plt.xlabel('Features')
            plt.ylabel('Value')
            plt.title('Line of best fit')
            plt.plot(X,np.dot(X, self.theta))
            plt.legend()
            plt.show()
        return self.theta

    def featureNormalize(self, X):
        normalizedX = np.zeros(self.X.shape)
        # X[0] gives first row of matrix
        for column in range(len(X[0])):
            mn = np.array(np.mean(X[:,column]))
            std = np.array(np.std(X[:,column]))
            self.mean[column] = mn
            self.std[column] = std
            a = X[:,column] - mn
            normalizedX[:,[column]] = np.divide(a,std)
        return normalizedX


    def normalEquation(self):
        self.theta = np.dot(np.dot(la.inv(np.dot(np.transpose(self.X), self.X)), np.transpose(self.X)),self.y)
        return self.theta

    def predict(self,x):
        normalizedX = np.divide((x - self.mean),self.std)
        return np.dot(normalizedX, self.theta)