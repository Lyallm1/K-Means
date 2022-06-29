import numpy as np, operator
from matplotlib import pyplot as plt, cm

r = lambda: np.random.randint(1, 100)

class Centroid:
    def __init__(self, pos):
        self.pos = pos
        self.points = self.previous_points = []
        self.color = None

class KMeans:
    def __init__(self, n_centroids=5):
        self.n_centroids = n_centroids
        self.centroids = []
        for _ in range(n_centroids): self.centroids.append(Centroid(np.array([r(), r()])))
        for i, c in enumerate(self.centroids): c.color = cm.rainbow(np.linspace(0, 1, len(self.centroids)))[i]

    def sample_data(self, samples=50):
        self.X = [[r(), r()] for _ in range(samples)]

    def fit(self):
        self.n_iters = 0
        fit = False 
        while not fit:
            for point in self.X: self.assign_centroid(point).points.append(point)
            if len([c for c in self.centroids if c.points == c.previous_points]) == self.n_centroids:
                fit = True
                self._update_centroids(reset=False)
            else: self._update_centroids()
            self.n_iters += 1

    def assign_centroid(self, x):
        distances = {}
        for centroid in self.centroids: distances[centroid] = np.linalg.norm(centroid.pos - x)
        return min(distances.items(), key=operator.itemgetter(1))[0]

    def _update_centroids(self, reset=True):
        for centroid in self.centroids:
            centroid.previous_points = centroid.points
            x_cor = [x[0] for x in centroid.points]
            y_cor = [y[1] for y in centroid.points]
            try: centroid.pos = [sum(x_cor) / len(x_cor), sum(y_cor) / len(y_cor)]
            except: pass
            if reset: centroid.points = []
        
    def show(self):
        for i, c in enumerate(self.centroids):
            plt.scatter(c.pos[0], c.pos[1], marker='o', color=c.color, s=75)
            plt.scatter([x[0] for x in c.points], [y[1] for y in c.points], marker='.', color=c.color)
        title = 'K-Means'
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()

if __name__ == '__main__':
    kmeans = KMeans()
    kmeans.sample_data()
    kmeans.fit()
    print(f'Iterations: {kmeans.n_iters}')
    kmeans.show()
