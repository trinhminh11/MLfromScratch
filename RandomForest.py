import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from DecisionTree import DecisionTree
from utils import most_common_value

random_seed = 0
np.random.seed(random_seed)

class RandomForest:
	def __init__(self, n_features, n_trees = 10, min_samples_split = 2, max_depth = 100):
		self.n_features = n_features
		self.n_trees = n_trees
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.root = None

		# array contain list of Tree
		self.trees: list[DecisionTree] = [None]*self.n_trees
	
	def fit(self, X: np.ndarray, y: np.ndarray):
		
		n_samples = X.shape[0]
		
		# build n_trees
		for i in range(self.n_trees):
			tree = DecisionTree(n_features=self.n_features, min_samples_split=self.min_samples_split, max_depth=self.max_depth)
			# samples
			idxs = np.random.choice(n_samples, n_samples, replace=True)
			tree.fit(X[idxs], y[idxs])

			self.trees[i] = tree
	
	def predict(self, X: np.ndarray):
		# pred for every Tree
		tree_preds = np.array([tree.predict(X) for tree in self.trees])

		# majority vote
		y_pred = np.array([most_common_value(pred) for pred in tree_preds.T])

		return y_pred
	
	def score(self, X, y):
		res = self.predict(X) == y
		return np.sum(res) / len(res)


def main():
	X, y = datasets.load_breast_cancer(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	model = RandomForest(X.shape[1], n_trees=5)

	model.fit(X_train, y_train)

	print(model.score(X_test, y_test))



if __name__ == "__main__":
	main()
