"""
 Authors : Fabien Robin and Mehdi Crozes
 Date : June 8th 2016
 Random Forest classifier for ARFF traffic network file
"""

import extraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

def main():
    """ Importing an arff file """

    data = extraction.load_dataset("../../Data/Caida/features_stats_best.arff")

    print "\tTotal dataset : "
    print "\tNumber of samples:", data.nb_examples
    print "\tNumber of features:", len(data.features[0])

    """ Splitting data into a training and testing set """

    feature_train, feature_test, label_train, label_test = extraction.kfold_data(data, 10)

    """param_grid = {"max_depth": [3, None],
                     "max_features": [1, 3, 8],
                     "min_samples_split": [1, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [True, False],
                     "criterion": ["gini", "entropy"]}"""

    #model = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid)
    model = RandomForestClassifier()
    model.fit(feature_train, label_train)

    predicted = model.predict(feature_test)

    print("Prediction : ", predicted, "\n")
    print("Labels : ", label_test, "\n")
    print("Score : ", model.score(feature_test, label_test))

if __name__ == '__main__':
    main()
