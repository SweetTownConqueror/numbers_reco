import pandas as pd
import matplotlib.pyplot as pt

from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('mnist_test.csv').as_matrix()

#train dataset
X = data[0:1000:, 1:]
y = data[0:1000, 0]
# X = data[0:10000:, 1:]
# y = data[0:10000, 0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)

my_classifier = KNeighborsClassifier(n_neighbors=3)
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

