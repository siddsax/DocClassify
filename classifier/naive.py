from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np

def classify_NB(train, test, method=None):
    
    if method is None:
        method = 'g'
    
    if method == 'b':
        classifier = BernoulliNB()
    elif method == 'm':
        classifier = MultinomialNB()
    else:
        classifier = GaussianNB()

    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]

    y_train_pred = classifier.fit(X_train, y_train).predict(X_train)
    y_test_pred = classifier.fit(X_train, y_train).predict(X_test)
    acc_train = (y_train_pred!=y_train).sum()/X_train.shape[0]
    acc_test = (y_test_pred!=y_test).sum()/X_test.shape[0]

    return acc_train, acc_test

