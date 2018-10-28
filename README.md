## Basic Document Representation and Classification

Dependencies 
```bash
Numpy
argparse
Sklearn
```
### DataSet
Download the large movie dataset from [here](http://ai.stanford.edu/~amaas/data/sentiment/) and place in the same folder 
named aclImdb.

Contains
### Representations
* Binary Bag of Words
* Normalized TF-IDF
* TF-IDF


### Classifiers
* Logistic Regression
* Naive Bayes
* SVM
* FeedForward Net

Running them 
```bash
python model.py [--n=#Number of training points to be used] [--C=#Classifier to be used.]
```
