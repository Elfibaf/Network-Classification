"""
Authors : Mehdi Crozes and Fabien Robin
Date : June 10th 2016
Update : June 20th 2016
LogisticClassifier with Weka's feature selection for ARFF traffic network file
"""

import time
from extraction import load_dataset, split_data, kfold_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

def main():

    """Importing Arff file with only 6 features"""

    #arff_file = load_dataset("../../Data/Caida/features_caida_flowcalc2.arff")
    arff_file = load_dataset_barcelona("../../Data/Info_file/packets_all_3.info")
    print "\tTotal dataset : "
    print "\tNumber of samples:", arff_file.nb_examples
    print "\tNumber of features:", len(arff_file.features[0])

    """Building training_set and test_set"""

    feature_train, feature_test, label_train, label_test = split_data(arff_file)
    #feature_train,feature_test,label_train,label_test = kfold_data(arff_file)

    """Modeling and training our classifier with training's time
    LogisticRegression parameters: solver to minimize loss function and multi_class='multinomial' for multi_class problem to use the sofmax function so as to find the predicted probability of each class"""

    clf = LogisticRegression(penalty='l2', max_iter=20, solver='newton-cg', multi_class='ovr')
    time_0 = time.time()
    clf.fit(feature_train, label_train)
    print "\tTraining time: ", round(time.time()-time_0, 3), "s"
    print "\tNumber of Classes :", len(clf.classes_)

    time_1 = time.time()
    label_pred = clf.predict(feature_test)
    print "\tPredicting time: ", round(time.time()-time_1, 3), "s"

    print "\tPrecision :", precision_score(label_test, label_pred, average='micro')
    print "\tRecall :", recall_score(label_test, label_pred, average='micro')

    print "\tTest Accuracy :", clf.score(feature_test, label_test)
    print "\tTraining Accuracy :", clf.score(feature_train, label_train)

if __name__ == '__main__':
    main()
