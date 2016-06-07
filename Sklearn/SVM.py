import re
import numpy as np
import os
import arff
from sklearn import svm
import extraction

model = svm.SVC()


data = extraction.load_dataset('../Data/Capture_Port.arff')
X = data.features
Y = data.labels


model.fit(X, Y)
print(model.score(X, Y))
