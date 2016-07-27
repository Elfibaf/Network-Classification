"""
Authors : Fabien Robin and Mehdi Crozes
Date : June 8th 2016
K Nearest Neighbours classifier for ARFF and INFO traffic network file
"""

import extraction
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score

def main():
    
    """ Main program """

    data = extraction.load_dataset("../../Data/Caida/Features_flowcalc/data_caida_original.arff")
    #data = extraction.load_dataset_barcelona("../../Data/Info_file/packets_all_3.info")
    print "\tTotal dataset : "
    print "\tNumber of samples:", data.nb_examples
    print "\tNumber of features:", len(data.features[0])

    #feature_train, feature_test, label_train, label_test = train_test_split(data.features, data.labels, test_size=0.25, random_state=42)
    feature_train, feature_test, label_train, label_test = extraction.kfold_data(data, 10)

    clf = KNeighborsClassifier()
    time_0 = time.time()
    clf.fit(feature_train, label_train)
    print "\tNumber of classes:", len(clf.classes_)
    print "\tTraining time ", round(time.time()-time_0, 3), "s"

    label_pred = clf.predict(feature_test)
    print "\tPrecision :", precision_score(label_test, label_pred, average='micro')
    print "\tRecall :", recall_score(label_test, label_pred, average='micro')

    print "\tTest Accuracy :", clf.score(feature_test, label_test)
    print "\tTrain Accuracy:", clf.score(feature_train, label_train)


if __name__ == '__main__':
    main()
