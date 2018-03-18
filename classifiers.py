from sklearn.linear_model import LogisticRegression
import numpy as np

class classifiers():
    
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def classify_lggistic(self, is_sparse=False):

        penalty = 'l1' if is_sparse else 'l2'
        classifier = LogisticRegression(penalty=penalty)

        X_train, y_train = self.train[0], self.train[1]
        X_test, y_test = self.test[0], self.test[1]

        y_train_pred = classifier.fit(X_train, y_train).predict(X_train)
        y_test_pred = classifier.fit(X_train, y_train).predict(X_test)
        acc_train = (y_train_pred != y_train).sum()*100.0/X_train.shape[0]
        acc_test = (y_test_pred != y_test).sum()*100.0/X_test.shape[0]

        return acc_train, acc_test
