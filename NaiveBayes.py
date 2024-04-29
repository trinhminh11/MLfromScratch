import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

n_samples = 1000
n_features = 10
n_classes = 2
random_seed = 0

class NaiveBayes:
	def __init__(self, n_features, n_classes) -> None:
		self.n_classes = n_classes
		self.n_features = n_features


		# mean to calc PDF
		self.mean = np.zeros((self.n_classes, n_features), dtype=np.float64)
		# variance to calc PDF
		self.var = np.zeros((self.n_classes, n_features), dtype=np.float64)

		self.priors = np.zeros(self.n_classes, dtype=np.float64)

	def fit(self, X: np.ndarray, y: np.ndarray):
		n_samples = X.shape[0]

		for c in range(self.n_classes):
			X_c = X[c==y]

			# mean of data y = c
			self.mean[c,:] = X_c.mean(axis=0)
			# variance of data y = c
			self.var[c,:] = X_c.var(axis=0)
			# prob of y = c
			self.priors[c] = X_c.shape[0] / n_samples

	# predict for many samples
	def predict(self, X: np.ndarray):
		return [self._predict(x) for x in X]

	# preidct for 1 samples
	def _predict(self, X: np.ndarray):

		res = -1
		max_post = float('-inf')

		for c in range(self.n_classes):
			prior = np.log(self.priors[c])
			# sum of log PDF
			class_cond = np.sum(np.log(self.PDF(c, X)))

			# Prob of where X = c
			post = prior + class_cond

			if post > max_post:
				max_post = post
				res = c
		
		return res

	# Probability Density Function
	def PDF(self, c, X):
		mean = self.mean[c]
		var = self.var[c]
		numerator = np.exp(-(X-mean)**2 / (2*var))
		denominator = np.sqrt(2 * np.pi * var)

		return numerator / denominator
	
	# calc Accuracy
	def score(self, X: np.ndarray, y: np.ndarray):
		res = self.predict(X) == y
		return np.sum(res) / len(res)

def main():
	X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_seed)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


	model = NaiveBayes(n_features, n_classes)

	model.fit(X_train, y_train)

	print(model.score(X_test, y_test))



if __name__ == "__main__":
	main()
