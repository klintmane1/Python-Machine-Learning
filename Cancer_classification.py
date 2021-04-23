# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:26:22 2021

Introduction to Machine Learning

For this project I will be using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients.

@author: klint
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Loading the data
cancer = load_breast_cancer()
print(cancer.DESCR) # Print the data set description
print(cancer.keys())

# Converting the data to a DataFrame from the bunch in scikit learn
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.DataFrame(cancer.target) # Adding a column with the labels

# Spliting the data
X = df.iloc[:,:30]
y = pd.Series(df['target']) # Creates a dataframe out of the target labels
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

# Creating the classifier object
knn = KNeighborsClassifier(n_neighbors = 1)

# Training the clasiffier
knn.fit(X_train, y_train)

# Using the knn classifier, we can predict the class label using the mean value for each feature
means = df.mean()[:-1].values.reshape(1, -1)
prediction_mean = knn.predict(means)

# Predictions for test sample
test_predictions = knn.predict(X_test)

# Mean accuracy 
mean_acc = knn.score(X_test, y_test)


#%% Plotting the results

import matplotlib.pyplot as plt

# Find the training and testing accuracies by target value (i.e. malignant, benign)
mal_train_X = X_train[y_train==0]
mal_train_y = y_train[y_train==0]
ben_train_X = X_train[y_train==1]
ben_train_y = y_train[y_train==1]

mal_test_X = X_test[y_test==0]
mal_test_y = y_test[y_test==0]
ben_test_X = X_test[y_test==1]
ben_test_y = y_test[y_test==1]


scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
          knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


plt.figure()

# Plot the scores as a bar chart
bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

# directly label the score onto the bars
for bar in bars:
    height = bar.get_height()
    plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                 ha='center', color='w', fontsize=11)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)