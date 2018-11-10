from __future__ import print_function
# import Trial
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# !pip
# install - q
# xgboost == 0.4
# a30
# from xgboost import XGBClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def preProcessing(df):
    #clean the data
    df = df.dropna()
    column_names = df.columns.values
    for i in column_names:
        df = df[~df[i].isin(['?'])]

    cols = len(df.columns)
    rows = len(df.index)
    #remove id
    x = df.iloc[:, 1:(cols - 2)].values
    y = df.iloc[:, (cols - 1)].values
    normalized_x = preprocessing.normalize(x)
    #label target values to class 1 and class 0
    for i in range(rows):
        if y[i] == 2:
            y[i] = 0
        if y[i] == 4:
            y[i] = 1
    # Split the dataset in two equal parts into 80:20 ratio for train:test
    x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test


def gridSearch(estimator, tuned_parameters):
    print(__doc__)
    file = open("Output.txt", "a+")
    # We are going to limit ourselves to accuracy score, other options can be
    # seen here:
    # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # Some other values used are the predcision_macro, recall_macro
    scores = ['accuracy']

    for score in scores:
        file.write("# Tuning hyper-parameters for %s\n" % score)
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(estimator, tuned_parameters, cv=5,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)
        file.write("Best parameters set found on development set:\n")
        print("Best parameters set found on development set:\n")
        print()
        file.write(str(clf.best_params_))
        print(clf.best_params_)
        print()

        file.write("\nGrid scores on development set:\n")
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            file.write("%0.3f (+/-%0.03f) for %r \n"
                       % (mean, std * 2, params))

        print()

        print("Detailed classification report:\n")
        file.write("Detailed classification report:\n")
        print()
        file.write("The model is trained on the full development set.\n")
        print("The model is trained on the full development set.")
        file.write("The scores are computed on the full evaluation set.\n")
        print("The scores are computed on the full evaluation set.\n")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        file.write(classification_report(y_true, y_pred) + "\n")
        print(classification_report(y_true, y_pred))
        file.write("Detailed confusion matrix:\n")
        print("Detailed confusion matrix:")
        file.write(str(confusion_matrix(y_true, y_pred)) + "\n")
        print(confusion_matrix(y_true, y_pred))
        file.write("Accuracy Score: \n")
        print("Accuracy Score: \n")
        file.write(str(accuracy_score(y_true, y_pred)) + "\n")
        print(accuracy_score(y_true, y_pred))

        print()

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin' \
           '.data'
df = pd.read_csv(path)
X_train, X_test, y_train, y_test = preProcessing(df)

tuningParameters = {DecisionTreeClassifier(): [{'max_depth': [None],
                                 'min_samples_split': [4, 6, 8],
                                 'min_samples_leaf': [2, 4, 6],
                                 'min_weight_fraction_leaf': [0],
                                 'max_features': ['auto'],
                                 'max_leaf_nodes': [None],
                                 'min_impurity_decrease': [0, 0.1]
                                 }],
                     MLPClassifier(): [{'activation': ['tanh', 'relu', 'identity', 'logistic'],
                                        'alpha': [0.001],
                                        'learning_rate': ['constant', 'adaptive', 'invscaling'],
                                        'max_iter': [200],
                                        'momentum': [0.89]}],

                     SVC(): [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4],
                              'C': [1, 10, 100, 1000]}
                             ],
                     GaussianNB(): [{'priors': [None, [0.55, 0.45]]}],

                     LogisticRegression(): [{
                                             'C': [0.8, 1.0, 1.2],
                                             'fit_intercept': [True, False],
                                             'class_weight': [None, 'balanced'],
                                             'solver': ['liblinear', 'saga'],
                                             'max_iter': [200, 150],
                                             'multi_class': ['ovr']
                                             }],
                     KNeighborsClassifier(): [{'n_neighbors': [1, 2, 3, 4, 5],
                                               'weights': ['uniform', 'distance'],
                                               'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                               'p': [1, 2, 3]
                                               }],
                     BaggingClassifier(): [{'n_estimators': [20],
                                            'max_samples': [0.6, 0.8, 1.0],
                                            'max_features': [0.6, 0.8, 1.0],
                                            'random_state': [None, 0]
                                            }],
                     RandomForestClassifier(): [{'n_estimators': [10, 20],
                                                 'criterion': ['gini', 'entropy'],
                                                 'max_depth': [None, 10],
                                                 'min_samples_split': [2, 4, 6, 8, 10],
                                                 'min_samples_leaf': [10, 0.1],
                                                 'max_features': [0.1, 'auto', 'log2', None]
                                                 }],
                     RandomForestClassifier(): [{'n_estimators': [10, 20],
                                                 'criterion': ['gini', 'entropy'],
                                                 'max_depth': [None, 10],
                                                 'min_samples_split': [2, 4, 6, 8, 10, 0.05],
                                                 'min_samples_leaf': [10, 0.1]
                                                 }],
                     AdaBoostClassifier(): [{'n_estimators': [50, 100, 200],
                                             'learning_rate': [0.6, 0.8, 0.9, 1.0],
                                             'algorithm': ['SAMME', 'SAMME.R'],
                                             'random_state': [None, 0]
                                             }],
                     GradientBoostingClassifier(): [{'loss': ['deviance', 'exponential'],
                                                     'learning_rate': [0.6, 0.8, 0.9, 1.0],
                                                     'n_estimators': [100, 150, 200],
                                                     'max_depth': [3, 5, 7, 9],
                                                     'min_samples_split': [2, 4, 6, 8, 10, 0.05],
                                                     'min_samples_leaf': [10, 0.1],
                                                     'min_impurity_decrease': [0, 0.1]
                                                     }],
                     # XGBClassifier(): [{'learning_rate': [0.1, 0.2, 0.3],
                     #                    'n_estimators': [100, 50, 150],
                     #                    'min_child_weight': [1, 2],
                     #                    'max_delta_step': [0, 1e-4],
                     #                    'seed': [0]
                     #                    }]

                     }

# c = {XGBClassifier(): [{'learning_rate': [0.1, 0.2, 0.3],
#                         'n_estimators': [100, 50, 150],
#                         'min_child_weight': [1, 2],
#                         'max_delta_step': [0, 1e-4],
#                         'seed': [0]
#                         }]}
file = open("Report.txt", "w").close()
for key, val in tuningParameters.items():
    file = open("Output.txt", "a+")
    file.write("Start Of Report-------------------------------------------------------------------\n")
    gridSearch(key, val)
    file.write("END of Report----------------------------------------------------------------------\n")