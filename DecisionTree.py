import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from utils import most_common_value

random_seed = 0
np.random.seed(random_seed)

class DecisionTree:

	class Node:
		# idx in data, threshold of node (<=threshold -> left, >threshold -> right), value of node (leafnode)
		def __init__(self, idx=None, threshold=None, left=None, right=None, value=None):
			self.idx = idx
			self.threshold = threshold
			self.left = left
			self.right = right
			self.value = value

		def isLeaf(self):
			return self.value is not None

	def __init__(self, n_features, min_samples_split = 2, max_depth = 100):
		self.n_features = n_features
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.root = None
	
	# calc information gain by calculating entropy
	def _information_gain(self, X, y, threshold):
		# parent Entropy
		parent_entropy = self.entropy(y)
		
		# spliting
		left, right = np.argwhere(X <= threshold).flatten(), np.argwhere(X > threshold).flatten()	

		n_l, n_r = len(left), len(right)

		# if cannot split into 2 children, information gain = 0
		if n_l & n_r == 0:	# if n_l == 0 or n_r == 0
			return 0
		
		# children Entropy
		n = len(y)
		e_l, e_r = self.entropy(y[left]), self.entropy(y[right])

		child_entropy = e_l * (n_l/n) + e_r * (n_r/n)

		# information gain
		return parent_entropy - child_entropy

	# select idx and threshold to split data
	def _best_criteria(self, X: np.ndarray, y: np.ndarray, idxs: np.ndarray):
		best_gain = -1
		split_idx, split_thres = None, None

		# loop through each index and threshold to find the best idx, threshold have the largest information gain
		for idx in idxs:
			X_column = X[:, idx]
			thresholds = np.unique(X_column)

			for threshold in thresholds:
				gain = self._information_gain(X_column, y, threshold)

				if gain > best_gain:
					best_gain = gain
					split_idx = idx
					split_thres = threshold
		
		return split_idx, split_thres

	# growing decision tree
	def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth = 0):
		n_samples, n_features = X.shape
		n_labels = len(np.unique(y))

		# at leaf node
		if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
			value = most_common_value(y)
			return self.Node(value=value)

		# get random indexs
		idxs = np.random.choice(n_features, self.n_features, replace=False)

		# greedy search
		idx, threshold = self._best_criteria(X, y, idxs)

		# select left indexes and right indexes base on best idx and threshold in X
		left_idxs, right_idxs = np.argwhere(X[: ,idx] <= threshold).flatten(), np.argwhere(X[:, idx] > threshold).flatten()

		# left node
		left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth=depth+1)
		# right node
		right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth=depth+1)

		return self.Node(idx, threshold, left, right)

	# fit is growing decision tree
	def fit(self, X: np.ndarray, y: np.ndarray):
		self.root = self._grow_tree(X, y)

	# traverse tree to predict
	def _traverse_tree(self, x, node: Node):
		if node.isLeaf():
			return node.value
		# got to left node
		if x[node.idx] <= node.threshold:
			return self._traverse_tree(x, node.left)
		# otherwise, right node
		return self._traverse_tree(x, node.right)

	def predict(self, X: np.ndarray):
		return np.array([self._traverse_tree(x, self.root) for x in X])
	
	# calculating entropy
	@staticmethod
	def entropy(y):
		hist = np.bincount(y)	#hist[i] = number occurent of i in y
		ps = hist/len(y)
		return -np.sum([p * np.log2(p) for p in ps if p > 0])
	
	def score(self, X, y):
		res = self.predict(X) == y
		return np.sum(res) / len(res)


def main():
	X, y = datasets.load_breast_cancer(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	model = DecisionTree(X.shape[1])

	model.fit(X_train, y_train)

	print(model.score(X_test, y_test))


if __name__ == "__main__":
	main()
