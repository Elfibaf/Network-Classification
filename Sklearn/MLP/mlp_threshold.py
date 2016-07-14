"""
Authors : Mehdi Crozes and Fabien Robin
Date : June 9th 2016
MLP for ARFF traffic network file with Threshold's feature selection
"""

import time

from extraction import load_dataset, load_dataset_barcelona, split_data, kfold_data
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

def main():

    """Importing Arff file """

    arff_file = load_dataset("../../Data/Caida/features_caida_flowcalc.arff")
    print "\tTotal dataset : "
    print "\tNumber of samples:", arff_file.nb_examples
    print "\tNumber of features:", len(arff_file.features[0])

    """ Building training_set and test_set"""

    feature_train, feature_test, label_train, label_test = split_data(arff_file)
    #feature_train, feature_test, label_train, label_test = kfold_data(arff_file, 10)

    """ Standardization + Variance Thresold of feature_train and feature_test"""

    scaler = MinMaxScaler(copy='false')
    feature_train_rescaled = scaler.fit_transform(feature_train)
    feature_test_rescaled = scaler.fit_transform(feature_test)

    selector = VarianceThreshold(threshold=0)
    feature_train_selected = selector.fit_transform(feature_train_rescaled, label_train)
    feature_test_selected = selector.fit_transform(feature_test_rescaled, label_test)
    print "\tNombre de features conserves :", len(feature_train_selected[1])
    print "\tListe des scores des labels:", selector.variances_

    """ Step 4 : Fitting and training our classifier
    Before that, we had to convert our feature_train_float into float type
    because our feature_train is an array of string and generate a TypeError"""

    time_0 = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(3, 500), max_iter=30, alpha=1e-5,
                        activation='relu', algorithm='sgd', verbose='true', tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    clf.fit(feature_train_selected, label_train)
    print "\tTraining  time ", round(time.time()-time_0, 3), "s"

    time_1 = time.time()
    label_pred = clf.predict(feature_test_selected)
    print "\tPredicting time ", round(time.time()-time_1, 3), "s"
    print "\tNumber of Classes :", len(clf.classes_)

    print "\tPrecision :", precision_score(label_test, label_pred, average='micro')
    print "\tRecall :", recall_score(label_test, label_pred, average='micro')

    clf.score(feature_train_selected, label_train)
    print "\tTest Accuracy :", clf.score(feature_test_selected, label_test)
    print "\tTraining Accuracy :", clf.score(feature_train_selected, label_train)

if __name__ == '__main__':
    main()
