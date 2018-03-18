from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
class classifiers():
    
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def results(self,classifier):
        y_train_pred = classifier.predict(self.train[0])
        y_test_pred = classifier.predict(self.test[0])
        acc_train = (y_train_pred == self.train[1]).sum()*100.0/self.train[0].shape[0]
        acc_test = (y_test_pred == self.test[1]).sum()*100.0/self.test[0].shape[0]

        return acc_train, acc_test
    
    def classify_lggistic(self):

        classifier = LogisticRegression().fit(self.train[0], self.train[1])
        return self.results(classifier)

    def naive_bayes(self):

        classifier = GaussianNB().fit(self.train[0], self.train[1])
        return self.results(classifier)

    def SVM(self):
        clf = svm.SVC(decision_function_shape='ovo')
        classifier = clf.fit(self.train[0], self.train[1])
        return self.results(classifier)
    
    def FFNet(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 50), random_state=1)
        classifier = clf.fit(self.train[0], self.train[1])
        return self.results(classifier)
