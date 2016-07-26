import numpy as np
import matplotlib.pyplot as plt
from math import e



class LogisticRegressionClassifier:
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.m = len(y)
        self.theta = np.zeros((self.X.shape[1],1))
        # mean and std need to be same dimensions as X
        self.mean = np.zeros(self.X.shape[0])
        self.std = np.zeros(self.X.shape[0])

    def gradientDescent(self,alpha,num_iters,featureNormalize=False,regularize=False,plot=True):
        cost_history = []
        if featureNormalize:
            X = self.featureNormalize(self.X)
        else:
            X = self.X
        if regularize:
            pass
        else:
            for i in range(num_iters):
                hypothesis = self.sigmoid(np.dot(X,self.theta))
                cost = (1/self.m)*sum((-self.y*np.log(hypothesis)) - ((1-self.y)*np.log(1-hypothesis)))
                cost_history.append(cost)
                print('Iteration #%d\tCost:%f' % (i, cost))
                gradient = np.transpose((1. / self.m) * np.transpose(self.sigmoid(X.dot(self.theta)) - self.y).dot(X))
                #gradient = np.zeros(self.theta.shape)
                #for i in range(len(self.theta)):
                #    gradient[i] = (1/self.m) * sum((hypothesis-self.y)*X[:,i:i+1])
                self.theta = self.theta - (alpha/self.m) * sum(gradient)
        if plot:
            plt.figure(0)
            plt.plot(cost_history, 'r')
            plt.ylabel('Cost')
            plt.xlabel('Iterations')
            plt.title('Cost function with Gradient Descent')
            plt.show()
        return self.theta

    def sigmoid(self,z):
        return 1 / (1+e**-z)

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

    def predict(self,x,normalize=False):
        if normalize:
            for iter in range(len(x)):
                x[iter] = x[iter] - self.mean[iter] / self.std[iter]
        hypothesis = self.sigmoid(np.dot(x,self.theta))
        if hypothesis >= .5:
            print("Class 1 with probability %f" % (hypothesis))
            return 1
        else:
            print("Class 0 with probability %f" % (1-hypothesis))
            return 0