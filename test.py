import networkx as nx
import pdb
from util import *

import torch
#
# x = torch.ones(2,2,2)
#
# print(x.size())
# print(x.repeat(4,2,1).size())
#
# pdb.set_trace()

# G = nx.DiGraph(nx.path_graph(4))
G = nx.grid_2d_graph(50,50)
G = nx.convert_node_labels_to_integers(G)
personalization = {}
for node in G.nodes():
    personalization[node] = 0
personalization[0]=1
t1 = time.time()
pr = nx.pagerank(G, alpha=0.9, personalization=personalization)
t2 = time.time()

pr_np = nx.pagerank_numpy(G, alpha=0.9, personalization=personalization)

t3 = time.time()
t4 = time.time()
print(t2-t1, t3-t2, t4-t3)

diff = 0
for i in range(len(pr)):
    diff += pr[i]-pr_np[i]
print(diff)

print(pr)
print(pr_np)

# plt.figure()
# nx.draw(G)
# plt.savefig('fig/view.png')