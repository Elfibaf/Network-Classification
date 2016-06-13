# Authors : Fabien Robin and Mehdi Crozes
# Date : June 8th 2016

# Random Forest classifier for ARFF traffic network file 

import re
import numpy as np
import os
import arff
import extraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


data = extraction.load_dataset('../../Data/Caida/data_caida_original.arff')
#data = extraction.load_dataset('../../Data/Capture_Port.arff')


# Splitting data into a training and testing set
feature_train,feature_test,label_train,label_test = train_test_split(data.features,data.labels,test_size=0.25,random_state=42)


param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 8],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

model = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid)
model.fit(feature_train, label_train)

predicted = model.predict(feature_test)

print("Prediction : ", predicted, "\n")
print("Labels : ", label_test, "\n")
print("Score : ", model.score(feature_test, label_test))
