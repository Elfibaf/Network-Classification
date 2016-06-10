import re
import numpy as np
import os
import arff
from sklearn import svm
import extraction
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.grid_search import GridSearchCV





#data = extraction.load_dataset('../../Data/Caida/data_caida_cut.arff')
data = extraction.load_dataset('../../Data/Capture_Port.arff')

#data.features = normalize(data.features)

# Splitting data into a training and testing set
feature_train,feature_test,label_train,label_test = train_test_split(data.features,data.labels,test_size=0.25,random_state=42)

#np.logspace(-20,0,num=21)
# Searching for best parameters for the classifier
"""param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [1e-12, 1e-10, 1e-13], 'kernel': ['rbf']},
 ]"""

#model = GridSearchCV(svm.SVC(C=1), param_grid)
model = svm.SVC(C = 1, gamma = 1e-12, cache_size = 3000)

X = feature_train
Y = label_train
model.fit(X, Y)
print(svm.SVC().get_params())

print("Best parameters : ", model.best_params_, "\n")

print("Model score with training data : ", model.score(X, Y))

predicted = model.predict(feature_test)
print("Classifier's prediction : ", predicted, "\n")
print("Score with testing data : ", model.score(feature_test, label_test))

