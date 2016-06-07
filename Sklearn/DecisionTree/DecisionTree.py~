import arff
import numpy as np
import matplotlib.pyplot as py

from extraction import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/Capture_Port.arff")

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)
print feature_train,feature_test,label_train,label_test

# Step 3 : Modeling and training our classifier

clf = DecisionTreeClassifier(random_state=0,max_depth=2)
clf.fit(feature_train,label_train)

# Step 4 : Accuracy 

score = clf.score(feature_test,label_test)
print "Accuracy :",score 

