import numpy as np 
from sklearn.datasets import load_svmlight_file

class representations():

    def __init__(self):
        self.test_loc = "aclImdb/test/labeledBow.feat"
        self.train_loc = "aclImdb/train/labeledBow.feat"
    
    def bow(self):
        data_test = load_svmlight_file(self.test_loc, n_features=89527)
        data_train = load_svmlight_file(self.train_loc, n_features=89527)
        
        return data_train, data_test

