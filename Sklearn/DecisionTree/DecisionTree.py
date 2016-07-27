"""
Authors : Mehdi Crozes and Fabien Robin
Date : June 7th 2016
Update : June 26th 2016
DecisionTreeClassifier for ARFF traffic network file
"""

import time
import matplotlib.pyplot as py
import numpy as np

from extraction import load_dataset, load_dataset_barcelona, plot_confusion_matrix, kfold_data, split_data
from sklearn import tree
from sklearn.metrics import confusion_matrix, recall_score, precision_score


def main():

    """ Main program """

    #arff_file = load_dataset("../../Data/Caida/Features_flowcalc/data_caida_original.arff")
    arff_file = load_dataset_barcelona('../../Data/Info_file/packets_all.info')

    print "\tTotal dataset : "
    print "\tNumber of samples:", arff_file.nb_examples
    print "\tNumber of features:", len(arff_file.features[0])

    """ Step 2: Building training_set and test_set """

    feature_train, feature_test, label_train, label_test = kfold_data(arff_file,10)

    """Step 3 : Fitting and training """

    clf = tree.DecisionTreeClassifier()
    t0 = time.time()
    clf.fit(feature_train, label_train)

    print "\tTraining time: ", round(time.time()-t0, 3), "s"
    print "\tNumber of Classes :", len(clf.classes_)
   
    t1 = time.time()
    label_pred = clf.predict(feature_test)
    print "\tPredicting time: ", round(time.time()-t1, 3), "s"

    """Step 4 : Precision_score,recall_score + plot confusion_matrix"""

    print "\tPrecision :", precision_score(label_test, label_pred, average='micro')
    print "\tRecall :", recall_score(label_test, label_pred, average='micro')

    cm = confusion_matrix(label_test, label_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    py.figure()
    plot_confusion_matrix(cm_normalized, clf)
    py.show()

    """Step 5 : Test and Train Accuracy"""

    score = clf.score(feature_test, label_test)
    score2 = clf.score(feature_train, label_train)
    print "\tTest Accuracy :", score
    print "\tTrain Accuracy:", score2

if __name__ == '__main__':
    main()
