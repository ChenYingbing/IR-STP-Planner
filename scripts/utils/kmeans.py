# from sklearn.cluster import KMeans
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from typing import List

class TrajectoryKmeans:
    """
    Implementation of k-means clustering algorithm. 
    These functions are designed to work with cartesian data points
    """
    def _convert_to_2d_array(self, points: np.ndarray):
        """
        Converts `points` to a 2-D numpy array.
        """
        points = np.array(points)
        if len(points.shape) == 1:
            points = np.expand_dims(points, -1)
        return points

    def SSE(self, points: np.ndarray):
        """
        Calculates the sum of squared errors for the given list of data points.
        Args:
            points: array-like
                Data points
        Returns:
            sse: float
                Sum of squared errors
        """
        points = self._convert_to_2d_array(points)
        centroid = np.mean(points, 0)
        e_norm = np.linalg.norm(centroid-points, 2, 2) # 10,12,2 -> 10,12
        
        batch_dist_means = np.mean(e_norm, axis=1)
        return np.mean(batch_dist_means), np.max(batch_dist_means)

        # e_norm = e_norm.sum(1)
        # return np.sum(e_norm), np.max(e_norm)

    def kmeans_max_se(self, points: np.ndarray, 
                      k:int=2, epochs:int=10, max_iter:int=100, 
                      verbose:bool=False, epsilon:float=20):
        """
        Clusters the list of points into `k` clusters using k-means clustering
        algorithm.
        Args:
            points: array-like
                Data points
            k: int
                Number of output clusters
            epochs: int
                Number of random starts (to find global optima)
            max_iter: int
                Max iteration per epoch
            verbose: bool
                Display progress on every iteration
        Returns:
            clusters: list with size = k
                List of clusters, where each cluster is a list of data points
        """
        points = self._convert_to_2d_array(points)
        assert len(points) >= k, "Number of data points can't be less than k"

        best_sse = np.inf
        best_max_se = np.inf
        for ep in range(epochs):
            # Randomly initialize k centroids
            np.random.shuffle(points)
            centroids = points[0:k, :]

            last_sse = np.inf
            last_max_se = np.inf
            for it in range(max_iter):
                # Cluster assignment
                clusters = [None] * k
                for p in points:
                    e_norm = np.linalg.norm(centroids-p, 2, 2) # 10,12,2 -> 10,12
                    e_norm = e_norm.sum(1)
                    index = np.argmin(e_norm)
                    if clusters[index] is None:
                        clusters[index] = np.expand_dims(p, 0)
                    else:
                        clusters[index] = np.vstack((clusters[index], np.expand_dims(p, 0)))

                # Check each cluster al least has one point
                clusters_clean = []
                for cluster in clusters:
                    if cluster is not None:
                        clusters_clean.append(cluster)

                # Centroid update
                # clusters = clusters[clusters != None]
                # try:
                centroids = [np.mean(c, 0) for c in clusters_clean]
                # except:
                #     print('numpy.AxisError: axis 0 is out of bounds for array of dimension 0')

                # SSE calculation
                sse = np.sum([self.SSE(c)[0] for c in clusters_clean])

                # MAX SE calculation
                max_sse = np.max([self.SSE(c)[1] for c in clusters_clean])

                gain_sse = last_sse - sse
                gain_max_sse = last_max_se - max_sse
                if verbose:
                    print('\rEpoch={}/Iter={}: '
                          'SSE:={:3f}, GainSSE:={:.3f}, '
                          'MaxSSE={:.3f}, GainMaxSSE: {:.4f};'.format(
                              ep, it, sse, gain_sse, max_sse, gain_max_sse
                          ), end="")

                # Check for improvement
                if sse <= best_sse and max_sse <= best_max_se:
                    best_clusters, best_sse, best_max_se = clusters, sse, max_sse

                # Epoch termination condition
                if np.isclose(gain_sse, 0, atol=0.00001) and max_sse <= epsilon:
                    break
                last_sse = sse
                last_max_se = max_sse

        return best_clusters, centroids, max_sse

    def kmeans(self, points: np.ndarray, k:int=2, epochs:int=10, max_iter:int=100, verbose=False):
        """
        Clusters the list of points into `k` clusters using k-means clustering
        algorithm.
        Args:
            points: array-like
                Data points
            k: int
                Number of output clusters
            epochs: int
                Number of random starts (to find global optima)
            max_iter: int
                Max iteration per epoch
            verbose: bool
                Display progress on every iteration
        Returns:
            clusters: list with size = k
                List of clusters, where each cluster is a list of data points
        """
        points = self._convert_to_2d_array(points)
        assert len(points) >= k, "Number of data points can't be less than k"

        best_sse = np.inf
        for ep in range(epochs):
            # Randomly initialize k centroids
            np.random.shuffle(points)
            centroids = points[0:k, :]

            last_sse = np.inf
            for it in range(max_iter):
                # Cluster assignment
                clusters = [None] * k
                for p in points:
                    e_norm = np.linalg.norm(centroids-p, 2, 2) # 10,12,2 -> 10,12
                    e_norm = e_norm.sum(1)
                    index = np.argmin(e_norm)
                    if clusters[index] is None:
                        clusters[index] = np.expand_dims(p, 0)
                    else:
                        clusters[index] = np.vstack((clusters[index], np.expand_dims(p, 0)))

                # Centroid update
                centroids = [np.mean(c, 0) for c in clusters]

                # SSE calculation
                sse = np.sum([self.SSE(c)[0] for c in clusters])

                # MAX SE calculation
                max_sse = np.max([self.SSE(c)[1] for c in clusters])

                gain = last_sse - sse
                if verbose:
                    print((f'Epoch: {ep:3d}, Iter: {it:4d}, '
                        f'SSE: {sse:12.4f}, Gain: {gain:12.4f}', f'Max SE: {max_sse:12.4f}'))

                # Check for improvement
                if sse < best_sse:
                    best_clusters, best_sse = clusters, sse

                # Epoch termination condition
                if np.isclose(gain, 0, atol=0.00001):
                    break
                last_sse = sse

        return best_clusters

    def bisecting_kmeans(self, points: np.ndarray, k: int=2, epochs: int=10, max_iter: int=100, verbose: bool=False):
        """
        Clusters the list of points into `k` clusters using bisecting k-means
        clustering algorithm. Internally, it uses the standard k-means with k=2 in
        each iteration.
        Args:
            points: array-like
                Data points
            k: int
                Number of output clusters
            epochs: int
                Number of random starts (to find global optima)
            max_iter: int
                Max iteration per epoch
            verbose: bool
                Display progress on every iteration
        Returns:
            clusters: list with size = k
                List of clusters, where each cluster is a list of data points
        """
        points = self._convert_to_2d_array(points)
        clusters = [points]
        while len(clusters) < k:
            max_sse_i = np.argmax([self.SSE(c) for c in clusters])
            cluster = clusters.pop(max_sse_i)
            two_clusters = self.kmeans(
                cluster, k=2, epochs=epochs, max_iter=max_iter, verbose=verbose)
            clusters.extend(two_clusters)
        return clusters

    def visualize_traj_clusters(self, clusters, centroids):
        """
        Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
        """
        plt.figure()
        for cluster in clusters:
            points = self._convert_to_2d_array(cluster)
            # if points.shape[1] < 2:
            #     points = np.hstack([points, np.zeros_like(points)])
            for traj in cluster:
                x = traj[:,0]
                y = traj[:,1]
                plt.scatter(x, y, color='b')
                plt.plot(x, y, color='b')

        for centroid in centroids:
            points = self._convert_to_2d_array(centroid)
            # if points.shape[1] < 2:
            #     points = np.hstack([points, np.zeros_like(points)])
            # for centroid in points:
            x = points[:,0]
            y = points[:,1]
            plt.scatter(x, y, color='y')
            plt.plot(x, y, color='y')

        plt.show()

    def visualize_clusters(self, clusters):
        """
        Visualizes the first 2 dimensions of the data as a 2-D scatter plot.
        """
        plt.figure()
        for cluster in clusters:
            points = self._convert_to_2d_array(cluster)
            if points.shape[1] < 2:
                points = np.hstack([points, np.zeros_like(points)])
            plt.plot(points[:,0], points[:,1], 'o')
        plt.show()
