# Importing the required packages
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def preData():
    path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin' \
           '.data'
    df = pd.read_csv(path)
    df = df.dropna()
    column_names = df.columns.values
    for i in column_names:
        df = df[~df[i].isin(['?'])]

    ncols = len(df.columns)
    nrows = len(df.index)
    x = df.iloc[:, 1:(ncols - 2)].values
    y = df.iloc[:, (ncols - 1)].values

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    normalized_x = preprocessing.normalize(x)

    for i in range(nrows):
        if y[i] == 2:
            y[i] = 0
        if y[i] == 4:
            y[i] = 1
    # Split the dataset in two equal parts into 80:20 ratio for train:test
    x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test
