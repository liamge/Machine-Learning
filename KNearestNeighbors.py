from sklearn import datasets
from math import sqrt

iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


class KNN:
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euc(self, v, w):
        assert len(v) == len(w)
        return sqrt(sum((v[i] - w[i]) ** 2 for i in range(len(v))))

    # Definitely a little messy, should try and make a more scalable solution
    def predict(self, observation, number_of_neighbors):
        distances = []
        for i in range(len(self.X_train)):
            distances.append([self.euc(observation, self.X_train[i]),i])
        distances = sorted(distances)
        nearest_neighbors = distances[:number_of_neighbors]
        targets = [self.y_train[v[1]] for v in nearest_neighbors]
        return max(targets, key=targets.count)

    def accuracy(self, answer_key, generated_answers):
        assert len(answer_key) == len(generated_answers)
        correct = 0
        for i in range(len(answer_key)):
            if answer_key[i] == generated_answers[i]:
                correct += 1
        return correct / len(answer_key)

knn = KNN()
knn.fit(X_train,y_train)
answers = []
for v in X_test:
    answers.append(knn.predict(v, 3))
print(knn.accuracy(y_test, answers))