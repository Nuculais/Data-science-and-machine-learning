# -*- coding: utf-8 -*-
"""
Hierarchial Clustering Practice
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

centers = [[1,1], [5,5], [3,10]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)

#plt.scatter(X[:,0], X[:,1])
#plt.show()

MS = MeanShift
MS.fit(X)
labels=MS.labels_
cluster_centers = MS.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters: ", n_clusters_)
colors= 10*['r.','g.','b.','c.','k.','y.','m.']

for i in range (len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 5)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
            marker="D", color='k', s=150, linewidths=5, zorder=10)
plt.show()
