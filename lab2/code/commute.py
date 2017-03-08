import numpy as np
import networkx as nx
import random

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster,linkage

G = nx.read_gml('dolphins.gml')
numpy_G = nx.to_numpy_matrix(G)

def  random_walk(start,end):
    commute_time = 0
    current_node = start
    while current_node != end:
        idx = random.randint(0,len(G.neighbors(current_node))-1)
        current_node = int(G.neighbors(current_node)[idx])
        commute_time+=1
    while current_node != start:
        idx = random.randint(0,len(G.neighbors(current_node))-1)
        current_node = int(G.neighbors(current_node)[idx])
        commute_time+=1
    return commute_time if commute_time!=0 else 0.0

commute_times = np.ones((G.number_of_nodes(),G.number_of_nodes()))

for i in range(0,G.number_of_nodes()-1):
    for j in range(0,G.number_of_nodes()-1):
        commute_times[i][j] = random_walk(i,j)

plt.matshow(commute_times, fignum=100, cmap=plt.cm.gray)
Z = linkage(commute_times, method = 'single', metric='euclidean')


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
clusters = fcluster(Z, 3000, 'distance')
print clusters
plt.show()

nx.draw_networkx(G, node_color=clusters)
plt.draw()
plt.show()

print("END")