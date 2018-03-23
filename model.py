import numpy as np
from classifiers import classifiers
from representations import representations


reps = representations()
data = reps.w2v(n=10)
rep = "w2v"

classifier = classifiers(data[0], data[1])
print(classifier.classify_lggistic())
if(rep == "glove" or rep == "w2v"):
    print(classifier.naive_bayes(flg=1))
else:
    print(classifier.naive_bayes())
print(classifier.SVM())
# print(classifier.FFNet())
