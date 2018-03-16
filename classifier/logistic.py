from sklearn.linear_model import LogisticRegression
import numpy as np

def classify_lggistic(train, test, is_sparse=False):

    penalty = 'l1' if is_sparse else 'l2'
    classifier = LogisticRegression(penalty=penalty)

    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]

    y_train_pred = classifier.fit(X_train, y_train).predict(X_train)
    y_test_pred = classifier.fit(X_train, y_train).predict(X_test)
    acc_train = (y_train_pred!=y_train).sum()/X_train.shape[0]
    acc_test = (y_test_pred!=y_test).sum()/X_test.shape[0]

    return acc_train, acc_test

