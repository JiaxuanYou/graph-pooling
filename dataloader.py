import torch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser

import pdb
import time
import random


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)





# def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
#     p = p_path
#     # path_count = max(int(np.ceil(p * k)),1)
#     path_count = max(int(np.ceil(p * k)),1)
#     G = nx.caveman_graph(c, k)
#     # remove 50% edges
#     p = 1-p_edge
#     for (u, v) in list(G.edges()):
#         if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
#             G.remove_edge(u, v)
#     # add path_count links
#     for i in range(path_count):
#         u = np.random.randint(0, k)
#         v = np.random.randint(k, k * 2)
#         G.add_edge(u, v)
#     G = max(nx.connected_component_subgraphs(G), key=len)
#     return G









def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'data/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs, data_node_att, data_node_label


# graphs = Graph_load_batch(name='PROTEINS_full')
# pdb.set_trace()


# def caveman_special(c=2,k=20,p_path=0.01):
#     G = nx.caveman_graph(c, k)
#     comps = [comp for comp in nx.connected_components(G)]
#
#     for edge in list(G.edges()):
#         if np.random.rand()<0.5:
#             G.remove_edge(edge[0],edge[1])
#
#     labels = {}
#     for id,comp in enumerate(comps):
#
#         for node in comp:
#             labels[node] = id
#
#     # pdb.set_trace()
#     for u in G.nodes():
#         for v in G.nodes():
#             if labels[u] != labels[v] and np.random.rand()<p_path:
#                 G.add_edge(u,v)
#
#     G = max(nx.connected_component_subgraphs(G), key=len)
#     print(G.number_of_nodes(), G.number_of_edges())
#     return G,labels


def caveman_special(l=2,k=20,p=0.1):
    G = nx.caveman_graph(l, k)
    comps = [comp for comp in nx.connected_components(G)]
    nodes = G.nodes()
    for (u, v) in G.edges():
        if random.random() < p:  # rewire the edge
            x = random.choice(nodes)
            if G.has_edge(u, x):
                continue
            G.remove_edge(u, v)
            G.add_edge(u, x)

    labels = {}
    for id,comp in enumerate(comps):
        for node in comp:
            labels[node] = id

    G = max(nx.connected_component_subgraphs(G), key=len)
    return G,labels
# caveman_special(c = 20, k = 20)

def load_graphs(dataset_str):
    if dataset_str == 'grid':
        graphs = []
        features = []
        for _ in range(1):
            graph = nx.grid_2d_graph(20, 20)
            # graph  = nx.grid_2d_graph(100, 100)
            graph = nx.convert_node_labels_to_integers(graph)

            # node_order = list(range(graph.number_of_nodes()))
            # shuffle(node_order)
            # order_mapping = dict(zip(graph.nodes(), node_order))
            # graph = nx.relabel_nodes(graph, order_mapping, copy=True)


            # feature = np.ones((graph.number_of_nodes(),1))
            feature = np.identity(graph.number_of_nodes())
            # label = nx.adjacency_matrix(graph).toarray()
            graphs.append(graph)
            features.append(feature)
        labels = None

    elif dataset_str == 'caveman_single':
        graph = nx.connected_caveman_graph(20, 20)
        feature = np.ones((graph.number_of_nodes(), 1))
        # feature = np.identity(graph.number_of_nodes())

        # graphs = [graph for _ in range(10)]
        # features = [feature for _ in range(10)]
        graphs = [graph]
        features = [feature]
        labels = None
        #
        # graphs = []
        # features = []
        # labels = None
        # for k in range(10):
        #     graphs.append(caveman_special(c=20, k=20, p_edge=0.2, p_path=500))
        #     features.append(np.ones((400, 1)))

    elif dataset_str == 'caveman':
        graphs = []
        features = []
        labels = []
        # labels = None
        for i in range(50):
            community_size = 20
            graph = nx.connected_caveman_graph(20, community_size)

            # graph,labels_dict = caveman_special(20,20,0)
            # node_dict = {}
            # for id, node in enumerate(graph.nodes()):
            #     node_dict[node] = id
            p=0.001
            count = 0
            for (u, v) in graph.edges():
                if random.random() < p:  # rewire the edge
                    x = random.choice(graph.nodes())
                    if graph.has_edge(u, x):
                        continue
                    graph.remove_edge(u, v)
                    graph.add_edge(u, x)
                    count += 1
            print('rewire:', count)



            n = graph.number_of_nodes()
            feature = np.ones((n, 1))
            label = np.zeros((n,n))
            for u in graph.nodes():
                for v in graph.nodes():
                    # if labels_dict[u] == labels_dict[v] and u!=v:
                    if u//community_size == v//community_size and u!=v:
                        label[u,v] = 1
                        # label[node_dict[u],node_dict[v]] = 1
            # feature = np.identity(graph.number_of_nodes())

            graphs.append(graph)
            features.append(feature)
            labels.append(label)

    elif dataset_str == 'protein':

        graphs_all, features_all, labels_all = Graph_load_batch(name='PROTEINS_full')
        features_all = (features_all-np.mean(features_all,axis=-1,keepdims=True))/np.std(features_all,axis=-1,keepdims=True)
        graphs = []
        features = []
        labels = []
        for graph in graphs_all:
            n = graph.number_of_nodes()
            label = np.zeros((n, n))
            for i,u in enumerate(graph.nodes()):
                for j,v in enumerate(graph.nodes()):
                    if labels_all[u-1] == labels_all[v-1] and u!=v:
                        label[i,j] = 1
            if label.sum() > n*n/2:
                continue

            graphs.append(graph)
            labels.append(label)

            idx = [node-1 for node in graph.nodes()]
            feature = features_all[idx,:]
            # label_dict = labels_all[graph.nodes()]
            features.append(feature)
            # pdb.set_trace()


        print('final num', len(graphs))


    elif dataset_str == 'email':

        with open('data/email.txt', 'rb') as f:
            graph = nx.read_edgelist(f)



        label_all = np.loadtxt('data/email_labels.txt')
        graph_label_all = label_all.copy()
        graph_label_all[:,1] = graph_label_all[:,1]//6


        for edge in graph.edges():
            if graph_label_all[int(edge[0])][1] != graph_label_all[int(edge[1])][1]:
                graph.remove_edge(edge[0], edge[1])

        comps = [comp for comp in nx.connected_components(graph) if len(comp)>10]
        graphs = [graph.subgraph(comp) for comp in comps]

        labels = []
        features = []

        for g in graphs:
            n = g.number_of_nodes()
            feature = np.ones((n, 1))
            features.append(feature)

            label = np.zeros((n, n))
            for i, u in enumerate(g.nodes()):
                for j, v in enumerate(g.nodes()):
                    if label_all[int(u)][1] == label_all[int(v)][1]:
                        label[i, j] = 1
            label = label - np.identity(n)
            labels.append(label)





    elif dataset_str == 'ppi':
        dataset_dir = 'data/ppi'
        print("Loading data...")
        G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
        labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
        labels = {int(i): l for i, l in labels.items()}

        train_ids = [n for n in G.nodes()]
        train_labels = np.array([labels[i] for i in train_ids])
        if train_labels.ndim == 1:
            train_labels = np.expand_dims(train_labels, 1)

        print("Using only features..")
        feats = np.load(dataset_dir + "/ppi-feats.npy")
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:, 0] = np.log(feats[:, 0] + 1.0)
        feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
        feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
        feat_id_map = {int(id): val for id, val in feat_id_map.items()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]

        # pdb.set_trace()

        node_dict = {}
        for id,node in enumerate(G.nodes()):
            node_dict[node] = id

        comps = [comp for comp in nx.connected_components(G) if len(comp)>10]
        graphs = [G.subgraph(comp) for comp in comps]

        id_all = []
        for comp in comps:
            id_temp = []
            for node in comp:
                id = node_dict[node]
                id_temp.append(id)
            id_all.append(np.array(id_temp))

        features = [train_feats[id_temp,:]+0.1 for id_temp in id_all]

        # graphs = [G.subgraph(comp) for comp in ]
        # pdb.set_trace()







    # real
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.from_dict_of_lists(graph)
        # keep the max connected component
        nodes_id = sorted(max(nx.connected_components(graph), key=len))
        graph = max(nx.connected_component_subgraphs(graph), key=len)
        # adj = nx.adjacency_matrix(G)

        feature = features[nodes_id, :].toarray()
        # feature = np.concatenate((np.identity(graph.number_of_nodes()), feature), axis=-1)


        graphs = [graph]
        features = [feature]
        labels = None

    return graphs, features, labels
# load cora, citeseer and pubmed dataset
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    # synthetic
    # todo: design node label
    labels, idx_train, idx_val, idx_test = None,None,None,None
    if dataset_str == 'grid':
        G = nx.grid_2d_graph(20, 20)
        # G = nx.grid_2d_graph(100, 100)
        # features = np.ones((G.number_of_nodes(),1))
        features = np.identity(G.number_of_nodes())
        labels = np.zeros((G.number_of_nodes(),2))
        labels[0:G.number_of_nodes()//2,0] = 1
        labels[G.number_of_nodes()//2:,1] = 1
        idx = np.random.permutation(G.number_of_nodes())
        idx_train = idx[0:G.number_of_nodes()//2]
        idx_val = idx[G.number_of_nodes()//2:]
    elif dataset_str == 'caveman':
        G = nx.connected_caveman_graph(20,20)
        features = np.identity(G.number_of_nodes())

        # features = np.ones((G.number_of_nodes(),1))
    elif dataset_str == 'barabasi':
        G = nx.barabasi_albert_graph(1000, 2)
        features = np.identity(G.number_of_nodes())

        # features = np.ones((G.number_of_nodes(), 1))

    # real
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                objects.append(pkl.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        G = nx.from_dict_of_lists(graph)
        # print(all(G.nodes()[i] <= G.nodes()[i + 1] for i in range(len(G.nodes()) - 1))) # check if sorted
        # keep the max connected component
        nodes_id = sorted(max(nx.connected_components(G), key=len))
        G = max(nx.connected_component_subgraphs(G), key=len)
        # adj = nx.adjacency_matrix(G)

        features = features[nodes_id,:]

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = labels[nodes_id,:]

        # idx_test = test_idx_range.tolist()
        # idx_train = range(len(y))
        # idx_val = range(len(y), len(y) + 500)
        idx_train = range(500)
        idx_val = range(500, 1000)
        idx_test = range(G.number_of_nodes()-1000,G.number_of_nodes())
    return G, features, labels, idx_train, idx_val, idx_test


    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])
    #
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]
    #
    # return G, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def get_random_subset(G, p=0.5, return_id = True):
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
    if return_id:
        nodes_return = [id for id,node in enumerate(nodes) if rand_values[id]<p]
    else:
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


def get_shortest_dist(shortest_dist, random_subsets):
    '''
    get the dist from a node to random subsets
    :param shortest_dist:
    :param node_id:
    :param random_subsets:
    :return: 2-d array, dist
    TODO: may consider different output format
    '''
    node_dist = np.zeros((1,len(random_subsets)))
    node_id = np.zeros((1,len(random_subsets)))
    for i, random_subset in enumerate(random_subsets):
        dist_min = 1e6 # todo: other aggregation possible: min, mean, sum, etc.
        node_min = 0
        for node in random_subset:
            dist = shortest_dist[node]
            if dist<dist_min:
                dist_min = dist
                node_min = node
        node_dist[0, i] = dist_min
        node_id[0, i] = node_min
    return node_dist, node_id

def get_shortest_dists(shortest_dists, random_subsets, nodes):
    '''
    get dist for all nodes
    :param shortest_dists:
    :param random_subsets:
    :param nodes: from G.nodes(), used to make sure node order is correct
    :return: subset_dists n*m, subset_ids n*m
    '''
    subset_dists = np.zeros((len(shortest_dists),len(random_subsets)))
    subset_ids = np.zeros((len(shortest_dists),len(random_subsets))).astype(int)
    for i,node_id in enumerate(nodes):
        shortest_dist = shortest_dists[node_id]
        node_dist, node_id_new = get_shortest_dist(shortest_dist,random_subsets)
        subset_dists[i] = node_dist
        subset_ids[i] = node_id_new
    return subset_dists, subset_ids



def get_feature(subset_ids, node_feature):
    '''
    match node ids for each subset with the corresponding features
    :param subset_ids: n*m
    :param node_feature: n*d
    :return: subset_features n*m*d
    '''
    # subset_features = np.zeros((subset_ids.shape[0],subset_ids.shape[1],
    #                             node_feature.shape[1]))
    # for i in range(subset_features.shape[0]):
    #     subset_features[i,:,:] = node_feature[subset_ids[i,:]]

    subset_features = node_feature[subset_ids.flatten(),:]
    subset_features = subset_features.reshape((subset_ids.shape[0],subset_ids.shape[1],
                                node_feature.shape[1]))

    return subset_features





class graph_dataset_node_classification(torch.utils.data.Dataset):
    def __init__(self, name = 'cora', permute = False):
        self.G, self.node_feature, self.label, self.idx_train, self.idx_val, self.idx_test = \
            load_data(name)
        self.n = self.G.number_of_nodes()
        self.subset_types = int(np.log2(self.G.number_of_nodes()))
        self.adj = nx.adjacency_matrix(self.G).toarray() + np.identity(self.n)
        try:
            self.node_feature = self.node_feature.toarray()
        except:
            pass
        self.node_feature = self.node_feature[:, np.newaxis, :]

        # G = max(nx.connected_component_subgraphs(G), key=len)
        self.G = nx.convert_node_labels_to_integers(self.G)

        self.node_label = np.zeros(self.label.shape[0])
        for i in range(self.label.shape[0]):
            self.node_label[i] = np.where(self.label[i] == 1)[0][0]
        self.num_class = self.label.shape[-1]

        self.shortest_dists = nx.shortest_path_length(self.G)

        self.permute = permute
        if not permute:
            self.recompute_feature()

    def recompute_feature(self):
        # compute dist
        t1 = time.time()
        self.random_subsets = get_random_subsets(self.G, c=0.5)
        t2 = time.time()
        self.subset_dists, self.subset_ids = get_shortest_dists(self.shortest_dists, self.random_subsets, self.G.nodes())
        t3 = time.time()
        self.subset_features = get_feature(self.subset_ids, self.node_feature[:,0,:]) # remove extra dim
        t4 = time.time()
        self.subset_dists = self.subset_dists[:, :, np.newaxis]

        t5 = time.time()
        print('node num:', self.G.number_of_nodes(), 'subset num:', len(self.random_subsets),
              'time:', t5 - t1, t2-t1,t3-t2,t4-t3,t5-t4)

        return self.subset_dists, self.subset_features


    def __len__(self):
        return self.subset_dists.shape[0]

    def __getitem__(self, idx):
        return self.node_feature[self.idx][idx], self.node_label[self.idx][idx], self.subset_dists[idx], self.subset_features[idx]

    def get_fullbatch_train(self):
        if self.permute:
            self.recompute_feature()
        return self.node_feature[self.idx_train], self.adj, self.node_label[self.idx_train], self.subset_dists[self.idx_train], self.subset_features[self.idx_train], self.subset_ids[self.idx_train]

    def get_fullbatch_val(self):
        if self.permute:
            self.recompute_feature()
        return self.node_feature[self.idx_val], self.adj, self.node_label[self.idx_val], self.subset_dists[self.idx_val], self.subset_features[self.idx_val], self.subset_ids[self.idx_val]

    def get_fullbatch_test(self):
        if self.permute:
            self.recompute_feature()
        return self.node_feature[self.idx_test], self.adj, self.node_label[self.idx_test], self.subset_dists[self.idx_test], self.subset_features[self.idx_test], self.subset_ids[self.idx_test]

    def get_fullbatch(self):
        if self.permute:
            self.recompute_feature()
        return self.node_feature, self.adj, self.node_label, self.subset_dists, self.subset_features, self.subset_ids





class graph_dataset_link_prediction(torch.utils.data.Dataset):
    def __init__(self, name = 'cora', test_ratio = 0.2, permute = False, approximate=False):
        self.G, self.node_feature, _, _, _, _ = load_data(name)
        self.n = self.G.number_of_nodes()
        self.subset_types = int(np.log2(self.G.number_of_nodes()))
        self.approximate = approximate

        # default value
        self.subset_dists, self.subset_features = np.zeros((0)), np.zeros((0))

        try:
            self.node_feature = self.node_feature.toarray()
        except:
            pass
        self.node_feature = self.node_feature[:, np.newaxis, :]

        self.G = nx.convert_node_labels_to_integers(self.G)

        self.split_dataset(test_ratio)
        assert self.G.nodes()==self.G_train.nodes()

        if approximate:
            self.node_dict = {}
            for i in range(self.n):
                self.node_dict[self.G.nodes()[i]] = i
        else:
            self.shortest_dists = nx.shortest_path_length(self.G_train)

        self.adj = nx.adjacency_matrix(self.G).toarray() + np.identity(self.n)
        self.adj_train = nx.adjacency_matrix(self.G_train).toarray() + np.identity(self.n)
        self.adj_test = self.adj - self.adj_train

        # mask
        num_positive_train = np.sum((self.adj_train>0).astype(int))
        self.mask_train = self.adj_train + np.random.rand(self.n, self.n)
        self.mask_train = (self.adj_train + (self.mask_train < num_positive_train/(self.n*self.n)).astype(int)).astype(bool).astype(int)
        num_positive_test = np.sum((self.adj_test>0).astype(int))
        self.mask_test = self.adj + np.random.rand(self.n, self.n)
        self.mask_test = (self.adj_test + (self.mask_test < num_positive_test / (self.n * self.n)).astype(int)).astype(bool).astype(int)

        self.permute = permute
        if not self.permute:
            self.recompute_feature()

    def split_dataset(self, test_ratio=0.2):
        self.G_train = self.G.copy()
        for edge in self.G_train.edges():
            self.G_train.remove_edge(edge[0],edge[1])
            if np.random.rand() > test_ratio or not nx.is_connected(self.G_train):
                self.G_train.add_edge(edge[0],edge[1])
        print('Train:', 'Connected', nx.is_connected(self.G_train),
              'Node', self.G_train.number_of_nodes(), 'Edge', self.G_train.number_of_edges())
        print('All:', 'Connected', nx.is_connected(self.G),
              'Node', self.G.number_of_nodes(), 'Edge', self.G.number_of_edges())


    def mask_adj_list(self):
        self.adj_list = self.G_train.adjacency_list()
        self.adj_count = np.zeros((self.n, self.n))
        # self.adj_count = np.zeros((len(self.random_subsets),self.n, self.n))

        # aggreagated adj_count
        for i,node_list in enumerate(self.adj_list):
            adj_list_temp = []
            for random_subset in self.random_subsets:
                node_list_temp = list(set(node_list) & set(random_subset))
                if len(node_list_temp)>0:
                    # adj_list_temp.append(node_list_temp)
                    adj_list_temp += node_list_temp
            for node in adj_list_temp:
                self.adj_count[i, self.node_dict[node]] += 1

        # batch version
        # for i,node_list in enumerate(self.adj_list):
        #     for b,random_subset in enumerate(self.random_subsets):
        #         node_list_temp = list(set(node_list) & set(random_subset))
        #         if len(node_list_temp)>0:
        #             for node in node_list_temp:
        #                 self.adj_count[b, i, self.node_dict[node]] += 1

        # pdb.set_trace()



    def recompute_feature(self):
        # compute dist
        t1 = time.time()
        self.random_subsets = get_random_subsets(self.G_train, c=1)

        if self.approximate:
            self.mask_adj_list()
        else:
            self.subset_dists, self.subset_ids = get_shortest_dists(self.shortest_dists, self.random_subsets, self.G_train.nodes())
            self.subset_features = get_feature(self.subset_ids, self.node_feature[:,0,:]) # remove extra dim
            self.subset_dists = self.subset_dists[:, :, np.newaxis]

        t2 = time.time()
        print('node num:', self.G_train.number_of_nodes(), 'subset num:', len(self.random_subsets),
              'time:', t2 - t1)

        # return self.subset_dists, self.subset_features

    def __len__(self):
        return self.G_train.number_of_nodes()

    def __getitem__(self, idx): # todo: edit for link pred
        return self.node_feature[self.idx][idx], self.subset_dists[idx], self.subset_features[idx]

    def get_fullbatch_train(self):
        if self.permute:
            self.recompute_feature()
        return (self.node_feature, self.adj_train, self.subset_dists, self.subset_features, self.mask_train)

    def get_fullbatch_test(self):
        if self.permute:
            self.recompute_feature()
        return (self.node_feature, self.adj_train, self.subset_dists, self.subset_features, self.mask_test, self.adj_test)





def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = []
    for node_adjaceny in A:
        num = 0
        for node in node_adjaceny:
            if node == 1.0:
                num = num + 1
        # Add an extra for the "self loop"
        num = num + 1
        degrees.append(num)
    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(degrees)
    # Cholesky decomposition of D
    D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Turn adjacency matrix into a numpy matrix
    A = np.matrix(A)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    return D @ A @ D
    # return A_hat, D




class graphs_dataset_loader():
    def __init__(self, name = 'grid', remove_link_ratio = 0.1, graph_test_ratio = 0.2,
                 permute = True, approximate=-1, normalize_adj = False):
        # args
        self.name = name
        self.remove_link_ratio = remove_link_ratio
        self.graph_test_ratio = graph_test_ratio
        self.permute = permute
        self.approximate = approximate
        self.normalize_adj = normalize_adj

        # 1 load data
        # list of networkx graphs; list of n*m arrays; list of n*n arrays/None(when link prediction)
        self.graphs, self.graphs_feature, self.graphs_label = load_graphs(self.name)

        # 2 (Link predition only) randomly remove edges for graphs, get different labels
        if self.remove_link_ratio>0:
            self.graphs, self.graphs_label_train, self.graphs_label_test = self.remove_link_graphs()
        else:
            self.graphs_label_train, self.graphs_label_test = self.graphs_label, self.graphs_label

        # 3 get adj
        self.graphs_adj = [nx.adjacency_matrix(graph).toarray() for graph in self.graphs]
        if self.normalize_adj:
            self.graphs_adj = [preprocess(adj) for adj in self.graphs_adj]

        # 4 precompute dists for all node pairs for all graphs
        self.graphs_dist = self.precompute_dist()

        # 5 set up mask for train and test
        self.graphs_mask_train, self.graphs_mask_test = self.set_masks()

        # 6 set up data index
        if len(self.graphs)>1:
            self.ids = np.random.permutation(len(self.graphs))
            self.ids_test = self.ids[:int(len(self.graphs) * self.graph_test_ratio)]
            self.ids_train = self.ids[int(len(self.graphs) * self.graph_test_ratio):]
        else: # transductive
            self.ids_test = np.array([0])
            self.ids_train = np.array([0])
        self.counter_train = 0
        self.counter_test = 0
        self.done_train = False
        self.done_test = False
        print(name, len(self.graphs))
        


    def set_masks(self):
        # for link prediction, two masks are different
        # for community detection, two masks are the same
        # Note: diag of adj should be 0!!!

        if self.remove_link_ratio > 0:
            graphs_mask_train = []
            graphs_mask_test = []
            for i in range(len(self.graphs)):
                adj = self.graphs_label_test[i]
                adj_train = self.graphs_label_train[i]
                adj_test = adj - adj_train
                n = adj_train.shape[0]

                num_positive_train = np.sum((adj_train > 0).astype(int))
                mask_train = adj_train + np.identity(n) + np.random.rand(n, n)
                mask_train = (adj_train + (mask_train < num_positive_train / (n * n)).astype(int)).astype(bool).astype(int)

                num_positive_test = np.sum((adj_test > 0).astype(int))
                mask_test = adj + np.identity(n) + np.random.rand(n, n)
                mask_test = (adj_test + (mask_test < num_positive_test / (n * n)).astype(int)).astype(bool).astype(int)

                graphs_mask_train.append(mask_train)
                graphs_mask_test.append(mask_test)

        else:
            graphs_mask_train = []
            for i in range(len(self.graphs)):
                adj = self.graphs_label_train[i]
                n = adj.shape[0]

                num_positive_train = np.sum((adj > 0).astype(int))
                mask_train = adj + np.identity(n) + np.random.rand(n, n)
                mask_train = (adj + (mask_train < num_positive_train / (n * n)).astype(int)).astype(bool).astype(int)

                graphs_mask_train.append(mask_train)
            graphs_mask_test = graphs_mask_train

        return graphs_mask_train, graphs_mask_test





    def get_batch_train(self):
        # reset epoch token
        if self.done_train:
            self.done_train = False

        id = self.ids_train[self.counter_train]

        self.counter_train += 1
        # check epoch ends
        if self.counter_train == len(self.ids_train):
            self.counter_train = 0
            self.done_train = True
            np.random.shuffle(self.ids_train)

        # re-sample random subsets
        self.random_subsets = get_random_subsets(self.graphs[id], c=1)
        self.dist_max, self.dist_argmax = self.get_shortest_dists(self.graphs_dist[id], self.random_subsets)

        return (self.graphs_adj[id], self.graphs_feature[id], self.graphs_dist[id], self.graphs_label_train[id], self.graphs_mask_train[id])

    def get_batch_test(self):
        # reset epoch token
        if self.done_test:
            self.done_test = False

        id = self.ids_test[self.counter_test]

        self.counter_test += 1
        # check epoch ends
        if self.counter_test == len(self.ids_test):
            self.counter_test = 0
            self.done_test = True
            np.random.shuffle(self.ids_test)

        # re-sample random subsets
        self.random_subsets = get_random_subsets(self.graphs[id], c=1)
        self.dist_max, self.dist_argmax = self.get_shortest_dists(self.graphs_dist[id], self.random_subsets)


        return (self.graphs_adj[id], self.graphs_feature[id], self.graphs_dist[id], self.graphs_label_test[id],
                self.graphs_mask_test[id])

    def precompute_dist(self):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graphs_dist = []
        for graph in self.graphs:
            if self.approximate>0:
                # dists_array = np.maximum(nx.adjacency_matrix(graph).toarray()*0.5 + np.identity(graph.number_of_nodes()), 0.1)
                # dists_array = nx.adjacency_matrix(graph).toarray()*0.5 + np.identity(graph.number_of_nodes())

                dists_array = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
                # todo: consider disconnected graph
                dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=self.approximate)
                for i, node_i in enumerate(graph.nodes()):
                    shortest_dist = dists_dict[node_i]
                    for j, node_j in enumerate(graph.nodes()):
                        dist = shortest_dist.get(node_j, -1)
                        if dist!=-1:
                            dists_array[i, j] = 1 / (dist + 1)

            else:
                dists_array = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
                # todo: consider disconnected graph
                dists_dict = nx.all_pairs_shortest_path_length(graph)
                for i, node_i in enumerate(graph.nodes()):
                    shortest_dist = dists_dict[node_i]
                    for j, node_j in enumerate(graph.nodes()):
                        dist = shortest_dist.get(node_j, -1)
                        if dist != -1:
                            dists_array[i, j] = 1 / (dist + 1)
            graphs_dist.append(dists_array)

        return graphs_dist

    def get_shortest_dists(self, graph_dist, random_subsets):
        dist_max = np.zeros((graph_dist.shape[0], len(random_subsets)))
        dist_argmax = np.zeros((graph_dist.shape[0], len(random_subsets)))
        for id,random_subset in enumerate(random_subsets):
            graph_dist_temp = graph_dist[:, random_subset]
            dist_max[:,id] = np.amax(graph_dist_temp, axis=-1)
            dist_argmax[:,id] = np.argmax(graph_dist_temp, axis=-1)

        return dist_max, dist_argmax



    def get_ordered_neighbours(self):
        pass



    def remove_link_graph(self, graph):
        graph_removed = graph.copy()
        for edge in graph_removed.edges():
            if np.random.rand() < self.remove_link_ratio:
                graph_removed.remove_edge(edge[0], edge[1])
                if self.name != 'ppi':
                    if not nx.is_connected(graph_removed):
                        graph_removed.add_edge(edge[0], edge[1])
        print('Before:', 'Connected', nx.is_connected(graph),
              'Node', graph.number_of_nodes(), 'Edge', graph.number_of_edges())
        print('After:', 'Connected', nx.is_connected(graph_removed),
              'Node', graph_removed.number_of_nodes(), 'Edge', graph_removed.number_of_edges())
        return graph_removed


    def remove_link_graphs(self):
        graphs_removed = []
        graphs_label_train = []
        graphs_label_test = []
        for graph in self.graphs:
            graph_removed = self.remove_link_graph(graph)
            graphs_removed.append(graph_removed)
            graphs_label_train.append(nx.adjacency_matrix(graph_removed).toarray())
            graphs_label_test.append(nx.adjacency_matrix(graph).toarray())
        return graphs_removed, graphs_label_train, graphs_label_test












def read_graphs():
    pass


# for explainer project
class graphs_dataset_loader_simple():
    def __init__(self, name='grid', remove_link_ratio=0.1, graph_test_ratio=0.2,
                 permute=True, approximate=-1, normalize_adj=False):
        # args
        self.name = name
        self.remove_link_ratio = remove_link_ratio
        self.graph_test_ratio = graph_test_ratio
        self.permute = permute
        self.approximate = approximate
        self.normalize_adj = normalize_adj

        # 1 load data
        # list of networkx graphs; list of n*m arrays; list of n*n arrays/None(when link prediction)
        self.graphs, self.graphs_feature, self.graphs_label = load_graphs(self.name)


        # 3 get adj
        self.graphs_adj = [nx.adjacency_matrix(graph).toarray() for graph in self.graphs]
        if self.normalize_adj:
            self.graphs_adj = [preprocess(adj) for adj in self.graphs_adj]


        # 6 set up data index
        self.counter_train = 0
        self.done_train = False
        print(name, len(self.graphs))



    def get_batch_train(self):
        # reset epoch token
        if self.done_train:
            self.done_train = False

        id = self.counter_train

        self.counter_train += 1
        # check epoch ends
        if self.counter_train == len(self.graphs):
            self.counter_train = 0
            self.done_train = True

        return (self.graphs_adj[id], self.graphs_feature[id])


# dataset = graphs_dataset_loader_simple()
# dataset.get_batch_train()


#
# t1 = time.time()
# dataset = graphs_dataset_loader(approximate=-1, name='ppi')
#
# for i in range(10):
#     t2 = time.time()
#     batch_train = dataset.get_batch_train()
#     t3 = time.time()
#     print(t3-t2)
# t2 = time.time()
# print(t2-t1)
# batch_test = dataset.get_batch_test()
# pdb.set_trace()
# dataset = graph_dataset_link_prediction(name='grid')





# 0113 archive
# class graph_dataset_link_prediction(torch.utils.data.Dataset):
#     def __init__(self, name = 'cora', test_ratio = 0.2, permute = False, approximate=False):
#         self.G, self.node_feature, _, _, _, _ = load_data(name)
#         self.n = self.G.number_of_nodes()
#         self.subset_types = int(np.log2(self.G.number_of_nodes()))
#
#         try:
#             self.node_feature = self.node_feature.toarray()
#         except:
#             pass
#         self.node_feature = self.node_feature[:, np.newaxis, :]
#
#         # G = max(nx.connected_component_subgraphs(G), key=len)
#
#         self.G = nx.convert_node_labels_to_integers(self.G)
#
#         self.node_dict = {}
#         for i in range(self.n):
#             self.node_dict[self.G.nodes()[i]] = i
#
#
#
#         self.split_dataset(test_ratio)
#         assert self.G.nodes()==self.G_train.nodes()
#
#         self.shortest_dists = nx.shortest_path_length(self.G_train)
#
#         # self.G_raw = self.G.copy()
#         # self.G_train_raw = self.G_train.copy()
#
#         # self.G = nx.convert_node_labels_to_integers(self.G)
#         # self.G_train = nx.convert_node_labels_to_integers(self.G_train)
#
#         self.adj = nx.adjacency_matrix(self.G).toarray() + np.identity(self.n)
#         self.adj_train = nx.adjacency_matrix(self.G_train).toarray() + np.identity(self.n)
#         self.adj_test = self.adj - self.adj_train
#
#         # mask
#         num_positive_train = np.sum((self.adj_train>0).astype(int))
#         self.mask_train = self.adj_train + np.random.rand(self.n, self.n)
#         self.mask_train = (self.adj_train + (self.mask_train < num_positive_train/(self.n*self.n)).astype(int)).astype(bool).astype(int)
#         num_positive_test = np.sum((self.adj_test>0).astype(int))
#         self.mask_test = self.adj + np.random.rand(self.n, self.n)
#         self.mask_test = (self.adj_test + (self.mask_test < num_positive_test / (self.n * self.n)).astype(int)).astype(bool).astype(int)
#
#         self.permute = permute
#         if not self.permute:
#             self.recompute_feature()
#
#     def split_dataset(self, test_ratio=0.2):
#         self.G_train = self.G.copy()
#         for edge in self.G_train.edges():
#             self.G_train.remove_edge(edge[0],edge[1])
#             if np.random.rand() > test_ratio or not nx.is_connected(self.G_train):
#                 self.G_train.add_edge(edge[0],edge[1])
#         print('Train:', 'Connected', nx.is_connected(self.G_train),
#               'Node', self.G_train.number_of_nodes(), 'Edge', self.G_train.number_of_edges())
#         print('All:', 'Connected', nx.is_connected(self.G),
#               'Node', self.G.number_of_nodes(), 'Edge', self.G.number_of_edges())
#
#     # def recompute_feature(self, G):
#     #     # compute dist
#     #     t1 = time.time()
#     #     # random_subsets = get_random_subsets(G, c=0.5)
#     #     random_subsets = get_random_subsets(G, c=1)
#     #     shortest_dists = nx.shortest_path_length(G)
#     #     subset_dists, subset_ids = get_shortest_dists(shortest_dists, random_subsets, G.nodes())
#     #     subset_features = get_feature(subset_ids, self.node_feature[:,0,:]) # remove extra dim
#     #
#     #     subset_dists = subset_dists[:, :, np.newaxis]
#     #
#     #     t2 = time.time()
#     #     print('node num:', self.G.number_of_nodes(), 'subset num:', len(random_subsets),
#     #           'time:', t2 - t1)
#     #     return subset_dists, subset_features
#
#     def mask_adj_list(self):
#         self.adj_list = self.G_train.adjacency_list()
#         self.adj_count = np.zeros((self.n, self.n))
#         # self.adj_count = np.zeros((len(self.random_subsets),self.n, self.n))
#
#         # aggreagated adj_count
#         for i,node_list in enumerate(self.adj_list):
#             adj_list_temp = []
#             for random_subset in self.random_subsets:
#                 node_list_temp = list(set(node_list) & set(random_subset))
#                 if len(node_list_temp)>0:
#                     # adj_list_temp.append(node_list_temp)
#                     adj_list_temp += node_list_temp
#             for node in adj_list_temp:
#                 self.adj_count[i, self.node_dict[node]] += 1
#
#
#         # for i,node_list in enumerate(self.adj_list):
#         #     for b,random_subset in enumerate(self.random_subsets):
#         #         node_list_temp = list(set(node_list) & set(random_subset))
#         #         if len(node_list_temp)>0:
#         #             for node in node_list_temp:
#         #                 self.adj_count[b, i, self.node_dict[node]] += 1
#
#         # pdb.set_trace()
#
#
#
#     def recompute_feature(self):
#         # compute dist
#         t1 = time.time()
#         self.random_subsets = get_random_subsets(self.G_train, c=1)
#         t2 = time.time()
#         self.subset_dists, self.subset_ids = get_shortest_dists(self.shortest_dists, self.random_subsets, self.G_train.nodes())
#         t3 = time.time()
#         self.subset_features = get_feature(self.subset_ids, self.node_feature[:,0,:]) # remove extra dim
#         t4 = time.time()
#         self.subset_dists = self.subset_dists[:, :, np.newaxis]
#
#         t5 = time.time()
#         print('node num:', self.G_train.number_of_nodes(), 'subset num:', len(self.random_subsets),
#               'time:', t5 - t1, t2-t1,t3-t2,t4-t3,t5-t4)
#
#         self.mask_adj_list()
#         return self.subset_dists, self.subset_features
#
#     def __len__(self):
#         return self.G_train.number_of_nodes()
#
#     def __getitem__(self, idx): # todo: edit for link pred
#         return self.node_feature[self.idx][idx], self.subset_dists[idx], self.subset_features[idx]
#
#     def get_fullbatch_train(self):
#         if self.permute:
#             self.recompute_feature()
#         return (self.node_feature, self.adj_train, self.subset_dists, self.subset_features, self.mask_train)
#
#     def get_fullbatch_test(self):
#         if self.permute:
#             self.recompute_feature()
#         return (self.node_feature, self.adj_train, self.subset_dists, self.subset_features, self.mask_test, self.adj_test)

