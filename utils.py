import numpy as np
from collections import Counter

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def most_common_value(y):
	return Counter(y).most_common(1)[0][0]
