import numpy as np 
from sklearn.datasets import load_svmlight_file


data = load_svmlight_file("aclImdb/test/labeledBow.feat")
print(data[0][1])
