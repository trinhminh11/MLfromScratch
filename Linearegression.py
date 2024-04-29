import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import datasets

n_samples = 100
n_features = 1
random_seed = 0

np.random.seed(random_seed)

class LinearRegression:
	def __init__(self, n_features) -> None:
		self.n_features = n_features

		# create random initial weights
		self.w = np.random.randn(n_features)

		# initial bias = 0
		self.b = 0

	def fit(self, X, y, epochs = 1000, lr = 0.0001):
		for epoch in tqdm(range(epochs)):
			dw, db = self.gradient_descend(X, y)

			# Applied Gradient
			self.w -= lr * dw
			self.b -= lr * db
	
	# gradient descend for Mean Squared Error
	def gradient_descend(self, X, y):
		n_samples = X.shape[0]

		y_predicted = self.predict(X)

		dw = (2/n_samples) * np.dot(X.T, (y_predicted - y))
		db = (2/n_samples) * np.sum(y_predicted-y)

		return dw, db

	# R Squared score
	def score(self, X: np.ndarray, y: np.ndarray):
		return 1 - self.MSE(X, y)/ (y.std() * y.std())
	
	# predict result
	def predict(self, X: np.ndarray):
		return np.dot(X, self.w) + self.b
	
	# Mean Squared Error
	def MSE(self, X: np.ndarray, y: np.ndarray):		
		return np.mean((y - self.predict(X))**2)

def main():
	X, y = datasets.make_regression(n_samples, n_features, noise = 20, random_state=random_seed)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	model = LinearRegression(X_train.shape[1])

	model.fit(X_train, y_train, lr = 0.01)

	print(model.score(X_test, y_test))


	# plot in first dims of X
	plt.scatter(X[:, 0], y, color = 'b', marker= 'o', s = 30)
	X_min = X[:, 0].min()
	X_max = X[:, 0].max()
	plt.plot([X_min, X_max], [model.predict(X[X[:, 0].min() == X[:, 0]]), model.predict(X[X[:, 0].max() == X[:, 0]])], c = 'red')
	plt.show()

	

if __name__ == "__main__":
	main()
