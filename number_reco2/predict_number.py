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



data = pd.read_csv('mnist_test.csv').as_matrix()

#train dataset
X = data[0:1000:, 1:]
y = data[0:1000, 0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .01)
#print(y_train)
#exit()
#training data
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

#test data
predictions = my_classifier.predict(X_test)
#predictions = my_classifier.predict(data[199:199:, 1:])

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
print(predictions[5])
#print(X_test[0])
d = X_test[5]
d.shape = (28, 28)
pt.imshow(255-d, cmap = 'gray')
pt.show()

