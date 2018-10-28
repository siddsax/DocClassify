import numpy as np
import argparse

from classifiers import classifiers
from representations import representations

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--N', dest='N', type=int, default=0, help='number of samples used for training. Default=0')
parser.add_argument('--C', dest='C', type=int, default=1, help='Classifier. 1=logistic, 2=Naive Bayes, 3=SVM')
args = parser.parse_args()

reps = representations()

if args.N != 0:
  data = reps.tfidf(n=args.N)
else:
  data = reps.tfidf(n)

classifier = classifiers(data[0], data[1])

if args.C == 1:
  print(classifier.classify_lggistic())
elif args.C == 2:
  print(classifier.naive_bayes())
else:
  print(classifier.SVM())
