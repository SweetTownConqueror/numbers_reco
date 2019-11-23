import pandas as pd
import matplotlib.pyplot as pt

from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
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
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

class Scrappy3NN():
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
        best_indices = []
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if len(best_indices) < 3:
                best_indices.append(i)
            else:
                longest_dist_test_train_indice = 0
                best_indice = 0
                longest_dist = -1;
                for j in range(len(best_indices)):
                    d1 = euc(row, self.X_train[best_indices[j]])
                    if d1 > longest_dist:
                        longest_dist = d1
                        longest_dist_test_train_indice = best_indices[j]
                        best_indice = j
                if dist < longest_dist:
                    best_indices[best_indice] = i
        if self.y_train[best_indices[0]] == self.y_train[best_indices[1]]:
            return self.y_train[best_indices[0]]
        elif self.y_train[best_indices[0]] == self.y_train[best_indices[2]]:
            return self.y_train[best_indices[0]]
        elif self.y_train[best_indices[1]] == self.y_train[best_indices[2]]:
            return self.y_train[best_indices[1]]
        else:
            dist1 = euc(row, self.X_train[best_indices[0]])
            dist2 = euc(row, self.X_train[best_indices[1]])
            dist3 = euc(row, self.X_train[best_indices[2]])
            smallest = dist1
            if (dist2 < dist1) and (dist2 < dist3):
                smallest = dist2
            elif (dist3 < dist1):
                smallest = dist3
            ret = 0
            if smallest == dist1:
                ret = self.y_train[best_indices[0]]
            if smallest == dist2:
                ret = self.y_train[best_indices[1]]
            if smallest == dist3:
                ret = self.y_train[best_indices[2]]
            return ret


data = pd.read_csv('mnist_test.csv').as_matrix()

#train dataset
X = data[0:1000:, 1:]
y = data[0:1000, 0]
# X = data[0:10000:, 1:]
# y = data[0:10000, 0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)

#training data
my_classifier = ScrappyKNN()
my_classifier = Scrappy3NN()
my_classifier.fit(X_train, y_train)

#test data
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
print(predictions[5])
d = X_test[5]
d.shape = (28, 28)
pt.imshow(255-d, cmap = 'gray')
pt.show()

