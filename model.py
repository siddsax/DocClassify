import numpy as np
from classifiers import classifiers
from representations import representations


reps = representations()
data = reps.tfidf(n=10)


classifier = classifiers(data[0], data[1])
print(classifier.classify_lggistic())
print(classifier.naive_bayes())
print(classifier.SVM())
# print(classifier.FFNet())
