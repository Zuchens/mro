from matplotlib.pyplot import draw
import networkx as nx
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage,cophenet
import numpy as np

import networkx as nx
G = nx.read_gml('dolphins.gml')
# nx.draw(G)
# plt.draw()
# plt.show()
numpy_G = nx.to_numpy_matrix(G)



# Z = linkage(numpy_G, method = 'ward', metric='cityblock')
Z = linkage(numpy_G, method = 'single', metric='euclidean')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

corelation_G = np.corrcoef(numpy_G)
Z_corr = linkage(corelation_G, method = 'ward', metric='euclidean')
c, coph_dists = cophenet(Z, pdist(numpy_G))

plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
d = dendrogram(
    Z,
    p=1000,
    count_sort='descending',
    distance_sort='ascending',
    truncate_mode= 'lastp',
    show_leaf_counts = True,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

# plt.show()
import community
import networkx as nx
import matplotlib.pyplot as plt

#better with karate_graph() as defined in networkx example.
#erdos renyi don't have true community structure
G = nx.erdos_renyi_graph(30, 0.05)

#first compute the best partition
partition = community.best_partition(G)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


nx.draw_networkx_edges(G,pos, alpha=0.5)
plt.show()

print "END"
