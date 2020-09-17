import numpy as np
from experiment.utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
import pdb


def CART_anomaly(dataset, a=12, b=1, c=2):

    dataset = normalize(dataset)
    for train, test in df_anomaly_label(dataset):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = DecisionTreeClassifier(max_depth=a, min_samples_leaf=b, min_samples_split=c)
    model.fit(train_input, train_output)
    test_predict = model.predict(test_input)
    test_actual = test_output.values
    # print(test_actual)
    # print(test_predict)
    tn, fp, fn, tp = my_confusion_matrix(test_actual, test_predict)
    # print(tn, fp, fn, tp)
    return confusion_matrix_calc(tn, fp, fn, tp)


def KNN_anomaly(dataset):
    n = len(dataset)
    if n < 6:
        n_neighbors = n - 1
    else:
        n_neighbors = 5

    dataset = normalize(dataset)
    for train, test in df_anomaly_label(dataset):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = neighbors.KNeighborsClassifier(n_neighbors)
    model.fit(train_input, train_output)
    test_predict = model.predict(test_input)
    test_actual = test_output.values
    # print(test_actual)
    # print(test_predict)
    tn, fp, fn, tp = my_confusion_matrix(test_actual, test_predict)
    # print(tn, fp, fn, tp)
    return confusion_matrix_calc(tn, fp, fn, tp)


def RFT_anomaly(dataset):

    dataset = normalize(dataset)
    for train, test in df_anomaly_label(dataset):
        train_input = train.iloc[:, :-1]
        train_output = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_output = test.iloc[:, -1]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_input, train_output)
    test_predict = model.predict(test_input)
    test_actual = test_output.values
    tn, fp, fn, tp = my_confusion_matrix(test_actual, test_predict)
    return confusion_matrix_calc(tn, fp, fn, tp)