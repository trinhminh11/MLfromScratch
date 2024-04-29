import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap
from math import dist
from collections import Counter

random_seed = 0

class KNN:
	def __init__(self, X, y, K = 3) -> None:
		self.K = K
		self.X = np.array(X)
		self.y = np.array(y)

	def predict(self, X):
		predicted_labels = [self._predict(x) for x in X]
		return np.array(predicted_labels)
	
	def _predict(self, X):
		distance = [dist(X, x) for x in self.X]
		k_indices = np.argsort(distance)[:self.K]

		# get k nearest values
		k_nearest = self.y[k_indices]
		
		# get the most common value in k_nearest
		most_common = Counter(k_nearest).most_common(1)[0][0]
		return most_common

	def score(self, X, y):
		res = self.predict(X) == y
		return np.sum(res) / len(res)

def main():
	X, y = datasets.load_iris(return_X_y=True)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	model = KNN(X_train, y_train)

	print(model.score(X_test, y_test))

	# scatter in first 2 dimensions
	plt.scatter(X[:, 0], X[:, 1], c = y, cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolors='k', s = 20)
	plt.show()

if __name__ == "__main__":
	main()
