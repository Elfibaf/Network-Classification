# Authors : Mehdi Crozes and Fabien Robin
# Date : June 7th 2016

# DecisionTreeClassifier for ARFF traffic network file 

import arff
import numpy as np
import matplotlib.pyplot as py
import time

from extraction import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/data_caida_original.arff")

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Modeling and training our classifier with training's time and parameters tune

t0=time.time()

clf = DecisionTreeClassifier()
clf.fit(feature_train,label_train)

print "Temps effectue ",round(time.time()-t0,3),"s"

#print clf.classes_ Print classes' list

# Step 4 : Accuracy 

score = clf.score(feature_test,label_test)
print "Accuracy :",score 

