import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

import pdb
import time



def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

def plot_graph(plt, G):
    plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    colors = []
    for i in range(len(values)):
        if values[i] == 0:
            colors.append('red')
        if values[i] == 1:
            colors.append('green')
        if values[i] == 2:
            colors.append('blue')
        if values[i] == 3:
            colors.append('yellow')
        if values[i] == 4:
            colors.append('orange')
        if values[i] == 5:
            colors.append('pink')
        if values[i] == 6:
            colors.append('black')
    plt.axis("off")
    pos = nx.spring_layout(G)
    # pos = nx.spectral_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=4, width=0.3, font_size = 3, node_color=colors,pos=pos)

def draw_graph_list(G_list, row, col, fname = 'figs/test'):
    # draw graph view
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plot_graph(plt, G)
        
    plt.tight_layout()
    plt.savefig(fname+'_view.png', dpi=600)
    plt.close()

    # draw degree distribution
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        G_deg = np.array(list(G.degree(G.nodes()).values()))
        bins = np.arange(20)
        plt.hist(np.array(G_deg), bins=bins, align='left')
        plt.xlabel('degree', fontsize = 3)
        plt.ylabel('count', fontsize = 3)
        G_deg_mean = 2*G.number_of_edges()/float(G.number_of_nodes())
        plt.title('average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=3)
        plt.tick_params(axis='both', which='minor', labelsize=3)
    plt.tight_layout()
    plt.savefig(fname+'_degree.png', dpi=600)
    plt.close()

    # degree_sequence = sorted(nx.degree(G).values(), reverse=True)  # degree sequence
    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()

    # draw clustering distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    G_cluster = list(nx.clustering(G).values())
    #    bins = np.linspace(0,1,20)
    #    plt.hist(np.array(G_cluster), bins=bins, align='left')
    #    plt.xlabel('clustering coefficient', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_cluster_mean = sum(G_cluster) / len(G_cluster)
    #    # if i % 2 == 0:
    #    #     plt.title('real average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    # else:
    #    #     plt.title('pred average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    plt.title('average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_clustering.png', dpi=600)
    #plt.close()

    ## draw circle distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    cycle_len = []
    #    cycle_all = nx.cycle_basis(G)
    #    for item in cycle_all:
    #        cycle_len.append(len(item))

    #    bins = np.arange(20)
    #    plt.hist(np.array(cycle_len), bins=bins, align='left')
    #    plt.xlabel('cycle length', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_cycle_mean = 0
    #    if len(cycle_len)>0:
    #        G_cycle_mean = sum(cycle_len) / len(cycle_len)
    #    # if i % 2 == 0:
    #    #     plt.title('real average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    # else:
    #    #     plt.title('pred average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    plt.title('average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_cycle.png', dpi=600)
    #plt.close()

    ## draw community distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    parts = community.best_partition(G)
    #    values = np.array([parts.get(node) for node in G.nodes()])
    #    counts = np.sort(np.bincount(values)[::-1])
    #    pos = np.arange(len(counts))
    #    plt.bar(pos,counts,align = 'edge')
    #    plt.xlabel('community ID', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_community_count = len(counts)
    #    # if i % 2 == 0:
    #    #     plt.title('real average clustering: {}'.format(G_community_count), fontsize=4)
    #    # else:
    #    #     plt.title('pred average clustering: {}'.format(G_community_count), fontsize=4)
    #    plt.title('average clustering: {}'.format(G_community_count), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_community.png', dpi=600)
    #plt.close()


def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a





# for GCN global

def get_random_subset(G, p=0.5):
    '''
    get a random subset of nodes
    :param G: input graph
    :param p: prob of including a node
    :return: a list of nodes, will not be empty
    '''
    nodes = G.nodes()
    while True:
        rand_values = np.random.rand(len(nodes))
        if np.any(np.less(rand_values,p)):
            break
    nodes_return = [node for id,node in enumerate(nodes) if rand_values[id]<p]
    return nodes_return

def get_random_subsets(G, c=1):
    '''
    get c*log^(n) random subsets of nodes
    :param G: input graph
    :param c: repeat same Sij for c*log(n) times
    :return: list of list of nodes, length fixed
    '''
    random_subsets = []
    for i in range(int(np.log2(G.number_of_nodes()))):
        p = 1/np.exp2(i+1)
        for j in range(int(np.log2(G.number_of_nodes())*c)):
            subset = get_random_subset(G,p)
            random_subsets.append(subset)
    return random_subsets


def get_shortest_dist(shortest_dist, node_id, random_subsets):
    '''
    get the dist from a node to random subsets
    :param shortest_dist:
    :param node_id:
    :param random_subsets:
    :return: 2-d array, dist
    TODO: may consider different output format
    '''
    node_feature = np.zeros((1,len(random_subsets)))
    for i, random_subset in enumerate(random_subsets):
        dist_min = 1e6 # todo: other aggregation possible: min, mean, sum, etc.
        for node in random_subset:
            dist = shortest_dist[node]
            if dist<dist_min:
                dist_min = dist
        node_feature[0, i] = dist_min
    return node_feature

def get_shortest_dists(shortest_dists, random_subsets):
    '''
    get dist for all nodes
    :param shortest_dists:
    :param random_subsets:
    :return: 2-d array, num_node * num_subsets
    '''
    node_features = np.zeros((len(shortest_dists),len(random_subsets)))
    for i,node_id in enumerate(shortest_dists):
        shortest_dist = shortest_dists[node_id]
        node_feature = get_shortest_dist(shortest_dist,node_id,random_subsets)
        node_features[i] = node_feature
    return node_features



# init graph
# G = nx.grid_2d_graph(20,20)
# G = nx.connected_caveman_graph(20,20)
# G = nx.barabasi_albert_graph(1000,2)
G = nx.newman_watts_strogatz_graph(200,2,0.1)



G = nx.convert_node_labels_to_integers(G)

# compute dist
t1 = time.time()
random_subsets = get_random_subsets(G,c=0.5)
shortest_dists = nx.shortest_path_length(G)
node_features = get_shortest_dists(shortest_dists, random_subsets)
t2 = time.time()
print('node num:', G.number_of_nodes())
print('subset num:', len(random_subsets))
print('time',t2-t1)


node_features_emb = TSNE(n_components=2).fit_transform(node_features)
print(node_features_emb.shape)

# plot results
plt.figure()
nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, node_size=4, width=0.3, font_size = 3)
# nx.draw_networkx(G, pos=nx.spectral_layout(G), with_labels=True, node_size=4, width=0.3, font_size = 3)
plt.savefig('fig/graph.png')
plt.close()

plt.figure()
plt.scatter(node_features_emb[:,0],node_features_emb[:,1])
plt.savefig('fig/emb.png')
plt.close()






