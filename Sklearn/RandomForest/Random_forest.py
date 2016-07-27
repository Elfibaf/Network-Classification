"""
 Authors : Fabien Robin and Mehdi Crozes
 Date : June 8th 2016
 Random Forest classifier for ARFF traffic network file
"""

import extraction
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def main():
    """ Main program """

    #data = extraction.load_dataset("../../Data/Caida/Features_flowcalc/data_caida_original.arff")
    data = extraction.load_dataset_barcelona('../../Data/Info_file/packets_all_3.info')

    print "\tTotal dataset : "
    print "\tNumber of samples:", data.nb_examples
    print "\tNumber of features:", len(data.features[0])

    """ Splitting data into a training and testing set """

    feature_train, feature_test, label_train, label_test = extraction.kfold_data(data, 10)

    model = RandomForestClassifier()
    time_0 = time.time()
    model.fit(feature_train, label_train)
    print "\tTraining time ", round(time.time()-time_0, 3), "s"

    time_1 = time.time()
    predicted = model.predict(feature_test)

    print "\tPredicting time ", round(time.time()-time_1, 3), "s"
    print "\tNumber of Classes :", len(model.classes_)

    print "\tPrecision :", precision_score(label_test, predicted, average='micro')
    print "\tRecall :", recall_score(label_test, predicted, average='micro')

    print "\tTest Accuracy :", model.score(feature_test, label_test)
    print "\tTraining Accuracy :", model.score(feature_train, label_train)

if __name__ == '__main__':
    main()
