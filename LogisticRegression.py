import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import datasets

from utils import sigmoid

random_seed = 0
np.random.seed(random_seed)


class LogisticRegression:
	def __init__(self, n_features) -> None:
		# create random initial weights
		self.w = np.random.randn(n_features)
		# initial bias = 0
		self.b = 0

	def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, lr=0.001):
		
		for epoch in tqdm(range(epochs)):
			dw, db = self.gradient_descend(X, y)

			# Applied Gradient
			self.w -= lr * dw
			self.b -= lr * db

	# gradient descent for Mean Squared Error + Sigmoid
	def gradient_descend(self, X: np.ndarray, y: np.ndarray):
		n_samples, n_features = X.shape

		linear_model = np.dot(X, self.w) + self.b
		y_predicted = sigmoid(linear_model)

		dw = (2/n_samples) * np.dot(X.T, (y_predicted-y))
		db = (2/n_samples) * np.sum(y_predicted-y)

		return dw, db

	# sigmoid > 0.5 = True, False otherwise
	def predict(self, X: np.ndarray):
		linear_model = np.dot(X, self.w) + self.b
		y_predicted = sigmoid(linear_model)

		return y_predicted >= 0.5
	
	# calc Accuracy
	def score(self, X, y):
		res = self.predict(X) == y
		return np.sum(res) / len(res)



def main():
	X, y = datasets.load_breast_cancer(return_X_y=True)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	model = LogisticRegression(X.shape[1])

	model.fit(X_train, y_train)

	print(model.score(X_test, y_test))



if __name__ == "__main__":
	main()
