import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap
from math import dist
from utils import most_common_value


n_center = 4
n_samples = 500
n_features = 2
cluster_std = 0.3

random_seed = 0
np.random.seed(random_seed)


class KMeansClustering:
	def __init__(self, K = 5) -> None:
		self.K = K

		# list contain K-Cluster
		self.cluster = [[] for _ in range(K)]
		
		# mean of each cluster
		self.centroids = np.array([None]*self.K)

	# finding closet centroid with given data x
	def _closest_centroid(self, x):
		res = float('inf')
		idx = -1
		for i in range(self.K):
			d = dist(self.centroids[i], x)
			if d < res:
				res = d
				idx = i

		return idx

	#  creating cluster with centroids
	def _create_cluster(self, X: np.ndarray):
		cluster = [[] for _ in range(self.K)]

		for i, x in enumerate(X):
			centroid_idx = self._closest_centroid(x)
			cluster[centroid_idx].append(i)

		return cluster
	
	# update new centroids = mean of cluster
	def _update_centroids(self, X):
		for i, cluster in enumerate(self.cluster):
			self.centroids[i] = np.mean(X[cluster], axis=0)

	# check if old centroids and the current centroids is the same
	def _is_converged(self, old):
		return np.allclose(old, self.centroids)

	def fit(self, X: np.ndarray, epochs = 100):
		n_samples = X.shape[0]

		# initialize centroid
		idxs = np.random.choice(n_samples, self.K, replace= False)
		self.centroids = X[idxs]
  
		# main loop
		for epoch in range(epochs):
			# update cluster
			self.cluster = self._create_cluster(X)

			# update centroid
			old_centroids = self.centroids.copy()
			self._update_centroids(X)

			# check if converge
			if self._is_converged(old_centroids):
				break
	
	# assign labels
	def predict(self, X: np.ndarray):
		n_samples = X.shape[0]
		labels = np.empty(n_samples)
		for i, x in enumerate(X):
			centroid_idx = self._closest_centroid(x)
			labels[i] = centroid_idx
		
		return labels
	
	# plot first 2 dims of X
	def plot(self, X):
		for cluster in self.cluster:
			point = X[cluster].T
			plt.scatter(*point[:2])
		
		for point in self.centroids:
			plt.scatter(*point[:2], marker = 'X', color = 'black', linewidths=2)

		plt.show()


def main():
	X, y = datasets.make_blobs(centers=n_center, n_samples=n_samples, n_features=n_features, cluster_std=cluster_std, shuffle=True, random_state=random_seed)


	model = KMeansClustering(n_center)

	model.fit(X)
	

	model.plot(X)



if __name__ == "__main__":
	main()
