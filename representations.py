import numpy as np 
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import binarize
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

class representations():

    def __init__(self):
        self.test_loc = "aclImdb/test/labeledBow.feat"
        self.train_loc = "aclImdb/train/labeledBow.feat"
        self.data_test = load_svmlight_file(self.test_loc, n_features=89527)
        self.data_train = load_svmlight_file(self.train_loc, n_features=89527)

    def binary_bow(self,n=None):
        data_test = self.data_test
        data_train = self.data_train
        
        if(n):
            X_tr = binarize(np.array(data_test[0][0:n].todense()))
            X_te = binarize(np.array(data_train[0][0:n].todense()))

            small_test = X_tr, data_test[1][0:n]
            small_train = X_te, data_train[1][0:n]
            
            return small_train, small_test

        return data_train, data_test

    def Ntf(self, n=None):
        data_test = self.data_test
        data_train = self.data_train
        if(n):
            X_tr = data_test[0][0:n].todense()
            X_te = data_train[0][0:n].todense()

            sum_tr = np.sum(X_tr,axis=1)
            sum_te = np.sum(X_te,axis=1)
            X_tr = X_tr/sum_tr
            X_te = X_te/sum_te

            small_test = X_tr, data_test[1][0:n]
            small_train = X_te, data_train[1][0:n]

            return small_train, small_test

        return data_train, data_test

    def tfidf(self, n=None):
        data_train,data_test = self.binary_bow(n)

        if(n):

            X_tr = data_train[0][0:n]
            X_te = data_test[0][0:n]

            idf_tr = np.log(np.shape(X_tr)[0]/np.sum(X_tr, axis=0))
            idf_te = np.log(np.shape(X_te)[0]/np.sum(X_te, axis=0))
            
            print(np.shape(X_tr)[0])
            print(np.sum(X_tr, axis=0))
            # X_tr = X_tr/sum_tr
            # X_te = X_te/sum_te
            where_are_NaNs = np.isinf(idf_te)
            idf_te[where_are_NaNs] = 0
            where_are_NaNs = np.isinf(idf_tr)
            idf_tr[where_are_NaNs] = 0
            
            tfidf_te = np.multiply(self.data_test[0][0:n].todense(),idf_te)
            tfidf_tr = np.multiply(self.data_train[0][0:n].todense(),idf_tr)

            print(self.data_test[0][0:n].todense())
            print(idf_te)
            print(tfidf_te)


            small_test = tfidf_te, data_test[1][0:n]
            small_train = tfidf_tr, data_train[1][0:n]

            return small_train, small_test

        return data_train, data_test

    def V_routine(self, model, n=None):
        f = open('aclImdb/imdb.vocab')
        vocab = f.read().splitlines()
        dims = len(model["cat"])
        word_vecs = np.zeros((len(vocab),dims))
        
        s_dim = 0
        for word in vocab:
            if word in model:
                word_vecs[s_dim] = model[word]
            s_dim+=1
        del model

        if(n):
            n_data_tr = n
            n_data_te = n
        else:
            n_data_tr = len(self.data_train)
            n_data_te = len(self.data_test)

        X_tr = self.data_train[0][0:n_data_tr].todense()
        X_te = self.data_test[0][0:n_data_te].todense()

        train_data = np.dot(X_tr, word_vecs)/np.sum(X_tr,axis=0), self.data_train[1][0:n_data_tr]
        test_data = np.dot(X_te, word_vecs)/np.sum(X_te,axis=0), self.data_test[1][0:n_data_te]

        return train_data,test_data

    def w2v(self,n=None):
        model = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

        return self.V_routine(model,n)

    def glove(self,n=None):
        glove_input_file = '~/Downloads/glove.6B.100d.txt'
        word2vec_output_file = '~/Downloads/glove.6B.100d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)
        print("-----")
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        print("======")
        return self.V_routine(model, n)
