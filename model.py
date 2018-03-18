import numpy as np
from classifiers import classifiers
from representations import representations


reps = representations()
data = reps.bow()

classifier = classifiers(data[0], data[1])
print(classifier.classify_lggistic(is_sparse=True))
