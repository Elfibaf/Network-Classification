"""
Authors : Fabien Robin and Mehdi Crozes
Date : June 9th 2016
SVM classifier for ARFF traffic network file
"""

from  extraction import load_dataset_barcelona, kfold_data
from sklearn import svm

def main():
    """Importing info file"""

    arff_file = load_dataset_barcelona("../../Data/Info_file/packets_all_1.info")
    print "\tTotal dataset : "
    print "\tNumber of samples:", arff_file.nb_examples
    print "\tNumber of features:", len(arff_file.features[0])

    feature_train, feature_test, label_train, label_test = kfold_data(arff_file, 10)

    model = svm.SVC(C=1, random_state=1)
    model.fit(feature_train, label_train)
    print(model.score(feature_train, label_train))

    predicted = model.predict(feature_test)
    print(model.score(feature_test, label_test))
    print(predicted)

if __name__ == '__main__':
    main()
