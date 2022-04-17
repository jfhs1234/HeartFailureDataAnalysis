import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sbn

# import heart failure dataset
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# creating input and output variables data

x = dataset.iloc[:, :-1]  # input variables
y = dataset.iloc[:, -1]  # output variable
# split data to training set and test set with assigned test size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# scaling features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# create new list to store accuracy scores
accuracy_score_list = []

# Logistic regression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)
y_prediction = classifier.predict(x_test)
a_s = accuracy_score(y_test, y_prediction)
accuracy_score_list.append(a_s)

#  Knn (find optimum neighbors number)
accuracy_list = []
for neighbors in range(1, 10):
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    accuracy_list.append(accuracy_score(y_test, y_prediction))

#  plot accuracy vs neighbors
plt.plot(list(range(1, 10)), accuracy_list)
plt.show()

# knn with best neighbor number
knn_best = np.argmax(accuracy_list)
classifier = KNeighborsClassifier(n_neighbors=(knn_best + 1))
classifier.fit(x_train, y_train)
y_prediction = classifier.predict(x_test)
a_s = accuracy_score(y_test, y_prediction)
accuracy_score_list.append(a_s)

# SVC (find optimum C)
accuracy_list = []
SVC_range = np.arange(0.1, 1, 0.1)
for c in SVC_range:
    classifier = SVC(C=c, random_state=0, kernel='rbf')
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    accuracy_list.append(accuracy_score(y_test, y_prediction))

plt.plot(SVC_range, accuracy_list)
plt.show()

# SVC with best c
SVC_best = SVC_range[np.argmax(accuracy_list)]
classifier = SVC(C=SVC_best, random_state=0, kernel='rbf')
classifier.fit(x_train, y_train)
y_prediction = classifier.predict(x_test)
a_s = accuracy_score(y_test, y_prediction)
accuracy_score_list.append(a_s)

#  tensorflow neural network
n_network = tf.keras.models.Sequential()

#  input layer with number of columns from x_train
n_network.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=np.shape(x_train)[1]))
n_network.add(tf.keras.layers.Dense(units=20, activation='relu'))
n_network.add(tf.keras.layers.Dense(units=20, activation='relu'))
n_network.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#  compile the network
n_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the network
n_network.fit(x_train, y_train, batch_size=32, epochs=100)

# predict and convert output values from fraction to 1 or 0
y_prediction = n_network.predict(x_test)
y_prediction = [0 if val < 0.5 else 1 for val in y_prediction]
a_s = accuracy_score(y_test, y_prediction)
accuracy_score_list.append(a_s)

# bar plot of accuracy scores
print(accuracy_score_list)
learning_list = ['Logistic Regression', 'Knn', 'SVC', 'Neural Network']
sbn.barplot(x=learning_list, y=accuracy_score_list)
plt.xlabel("Model", fontsize=10)
plt.ylabel("% Accuracy", fontsize=10)
plt.title('Bar plot of model accuracy')
plt.show()

