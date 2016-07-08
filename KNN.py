# K Nearest Neighbors based upon Google's Machine Learning Recipies playlist on Youtube
from sklearn import datasets
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a,b)

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Meat and potatoes of the classifier goes here
class StippedDownKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

classifier = StippedDownKNN()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
average_accuracy = [accuracy_score(y_test, classifier.predict(X_test)) for i in range(25)]
print(sum(average_accuracy)/len(average_accuracy))