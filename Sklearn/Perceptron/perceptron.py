"""
Authors : Mehdi Crozes and Fabien Robin
Date : June 8th 2016
Perceptron classifier for ARFF traffic network file
"""

import time

from extraction import load_dataset, split_data, kfold_data
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

def main():

    """ Importing Arff file """

    arff_file = load_dataset("../../Data/Caida/Features_flowcalc/features_stats_best.arff")
    print "\tTotal dataset : "
    print "\tNumber of samples:", arff_file.nb_examples
    print "\tNumber of features:", len(arff_file.features[0])

    """ Building training_set and test_set (Kfold or split) """

    #feature_train,feature_test,label_train,label_test = split_data(arff_file)
    feature_train, feature_test, label_train, label_test = kfold_data(arff_file, 10)

    """Fitting and training our classifier """

    scaler = MinMaxScaler(copy='false')
    feature_train_rescaled = scaler.fit_transform(feature_train)
    feature_test_rescaled = scaler.fit_transform(feature_test)

    clf = Perceptron(n_iter=30)
    time_0 = time.time()
    clf.fit(feature_train_rescaled, label_train)
    print "\tTraining time ", round(time.time()-time_0, 3), "s"

    time_1 = time.time()
    label_pred = clf.predict(feature_test_rescaled)
    print "\tPredicting time ", round(time.time()-time_1, 3), "s"
    print "\tNumber of Classes :", len(clf.classes_)

    print "\tPrecision :", precision_score(label_test, label_pred, average='micro')
    print "\tRecall :", recall_score(label_test, label_pred, average='micro')

    print "\tTest Accuracy :", clf.score(feature_test_rescaled, label_test)
    print "\tTraining Accuracy :", clf.score(feature_train_rescaled, label_train)

if __name__ == '__main__':
    main()
