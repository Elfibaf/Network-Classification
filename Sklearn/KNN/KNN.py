import re
import numpy as np
import os
import arff
from sklearn.neighbors import KNeighborsClassifier
import extraction
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


data = extraction.load_dataset('../../Data/Caida/data_caida_original.arff')
#data = extraction.load_dataset('../../Data/Capture_Port.arff')


# Splitting data into a training and testing set
feature_train,feature_test,label_train,label_test = train_test_split(data.features,data.labels,test_size=0.25,random_state=42)

"""param_grid = [
  {'leaf_size': [30, 20, 10, 40, 50], 'n_neighbors' : [5, 6, 7]},
 ]"""


#model = GridSearchCV(KNeighborsClassifier(), param_grid)
model = KNeighborsClassifier()
model.fit(feature_train, label_train)
print(len(model.classes_))

#print("Best params : ", model.best_params_)

predicted = model.predict(feature_test)

print("Prediction : ", predicted, "\n")
print("Labels : ", label_test, "\n")
print("Score : ", model.score(feature_test, label_test))
