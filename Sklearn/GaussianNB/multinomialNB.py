"""
 Authors : Mehdi Crozes and Fabien Robin
 Date : June 7th 2016
 Update : June 22nd 2016
 MultinomialNB with feature selection weka for ARFF traffic network file
"""

import time
import math

from extraction import load_dataset, load_dataset_barcelona, split_data, kfold_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler

def main():

    """ Importing Arff file """

    arff_file = load_dataset("../../Data/Caida/Features_flowcalc/data_caida_original.arff")
    #arff_file=load_dataset_barcelona("../../Data/Info_file/packets_all_1.info")
    print "Total dataset : "
    print "\tNumber of samples:", arff_file.nb_examples
    print "\tNumber of features:", len(arff_file.features[0])

    """ Building training_set and test_set """

    #feature_train, feature_test, label_train, label_test = split_data(arff_file)
    feature_train, feature_test, label_train, label_test = kfold_data(arff_file, 10)

    """Fitting and training our classifier """

    scaler = MinMaxScaler(copy='false')
    feature_train_rescaled = scaler.fit_transform(feature_train)
    feature_test_rescaled = scaler.fit_transform(feature_test)

    clf = MultinomialNB()
    time_0 = time.time()
    clf.fit(feature_train_rescaled, label_train)
    print "\tTraining time ", round(time.time()-time_0, 3), "s"

    log_prob_class = clf.class_log_prior_
    prob_class = [math.pow(10, log_prob_class[i]) for i in log_prob_class]
    print "\tMax probablilite ", round(max(prob_class), 10)

    time_1 = time.time()
    label_pred = clf.predict(feature_test_rescaled)
    print "\tPredicting time: ", round(time.time()-time_1, 3), "s"

    """ Precision_score and recall_score"""

    print "\tPrecision :", precision_score(label_test, label_pred, average='micro')
    print "\tRecall :", recall_score(label_test, label_pred, average='micro')

    print "\tTest Accuracy :", clf.score(feature_test_rescaled, label_test)
    print "\tTrain Accuracy:", clf.score(feature_train_rescaled, label_train)

if __name__ == '__main__':
    main()
