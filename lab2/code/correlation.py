import numpy as np
import networkx as nx

G = nx.read_gml('dolphins.gml')
numpy_G = nx.to_numpy_matrix(G)
corelation_G = np.corrcoef(numpy_G)
print corelation_G
import seaborn as sns
import pandas as pd
data  = pd.DataFrame(data=corelation_G, index=[x for x in range(0,62)], columns=[x for x in range(0,62)])
sns.clustermap(data=data)
sns.plt.show()