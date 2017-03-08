import pickle
import random
from matplotlib.pyplot import draw

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,cophenet, fcluster
import numpy as np
import networkx as nx

def shortest(G):
    shortest_paths = np.zeros((G.number_of_nodes(),G.number_of_nodes()))
    for i in range(0,G.number_of_nodes()-1):
        for j in range(0,G.number_of_nodes()-1):
            shortest_paths[i][j] = nx.shortest_path_length(G,source=i,target=j)

    return shortest_paths

def correlation(numpy_G):
    return np.corrcoef(numpy_G)
G = nx.read_gml('dolphins.gml')
numpy_G = nx.to_numpy_matrix(G)


# Z = linkage(numpy_G, method = 'ward', metric='cityblock')
Z = linkage(numpy_G, method = 'single', metric='euclidean')
# Z = linkage(numpy_G, method = 'complete', metric='euclidean')
# Z = linkage(numpy_G, method = 'average', metric='euclidean')

plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
d = dendrogram(
    Z,
    # p=60,
    show_contracted=True,
    count_sort='descending',
    distance_sort='ascending',
    # truncate_mode= 'lastp',
    show_leaf_counts = True,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
clusters = fcluster(Z, 2.5, 'distance')
print clusters
plt.show()

nx.draw_networkx(G, node_color=clusters)
plt.draw()
plt.show()

print("END")