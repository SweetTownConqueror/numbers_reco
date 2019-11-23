import pandas as pd
import matplotlib.pyplot as pt

from scipy.spatial import distance
from sklearn import svm
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('mnist_test.csv').as_matrix()

#train dataset
X = data[0:10000:, 1:]
y = data[0:10000, 0]
# X = data[0:1000:, 1:]
# y = data[0:1000, 0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)
#use a model
#not good
#my_classifier = svm.SVC(kernel='rbf', C=1.0)
#good
#my_classifier = svm.SVC(kernel='poly', C=1.0)
#too long my_classifier = svm.SVC(kernel='sigmoid', C=1.0)
my_classifier = svm.SVC(kernel='linear', C=1.0)
#my_classifier = svm.SVC()
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

