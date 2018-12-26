import torch
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data


import pdb
import time


# for GCN global
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
        node_dist, node_id = get_shortest_dist(shortest_dist,random_subsets)
        subset_dists[i] = node_dist
        subset_ids[i] = node_id
    return subset_dists, subset_ids


def get_feature(subset_ids, node_feature):
    '''
    match node ids for each subset with the corresponding features
    :param subset_ids: n*m
    :param node_feature: n*d
    :return: subset_features n*m*d
    '''
    subset_features = np.zeros((subset_ids.shape[0],subset_ids.shape[1],
                                node_feature.shape[1]))
    for i in range(subset_features.shape[0]):
        subset_features[i,:,:] = node_feature[subset_ids[i,:]]
    return subset_features





class graph_dataset_node_classification(torch.utils.data.Dataset):
    def __init__(self, name = 'cora', type = 'train'):
        self.G, self.node_feature, self.label, self.idx_train, self.idx_val, self.idx_test = \
            load_data(name)
        self.adj = nx.adjacency_matrix(self.G)
        self.node_feature = self.node_feature.toarray()
        self.node_feature = self.node_feature[:, np.newaxis, :]

        # G = max(nx.connected_component_subgraphs(G), key=len)
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.node_label = np.zeros(self.label.shape[0])
        for i in range(self.label.shape[0]):
            self.node_label[i] = np.where(self.label[i] == 1)[0][0]
        self.num_class = self.label.shape[-1]

        pdb.set_trace()
        # recompute feature
        if type=='train':
            self.idx = self.idx_train
        if type=='val':
            self.idx = self.idx_val
        if type=='test':
            self.idx = self.idx_test
        self.recompute_feature()

    def recompute_feature(self):
        # compute dist
        t1 = time.time()
        self.random_subsets = get_random_subsets(self.G, c=0.5)
        self.shortest_dists = nx.shortest_path_length(self.G)
        self.subset_dists, self.subset_ids = get_shortest_dists(self.shortest_dists, self.random_subsets, self.G.nodes())
        self.subset_features = get_feature(self.subset_ids, self.node_feature[:,0,:]) # remove extra dim

        self.subset_dists = self.subset_dists[:, :, np.newaxis]

        self.subset_dists = self.subset_dists[self.idx]
        self.subset_features = self.subset_features[self.idx]


        t2 = time.time()
        print('node num:', self.G.number_of_nodes(), 'subset num:', len(self.random_subsets),
              'time:', t2 - t1)


    def __len__(self):
        return self.subset_dists.shape[0]

    def __getitem__(self, idx):
        return self.node_feature[self.idx][idx], self.node_label[self.idx][idx], self.subset_dists[idx], self.subset_features[idx]

    def get_fullbatch(self):
        return self.node_feature, self.node_label, self.subset_dists, self.subset_features








class graph_dataset_link_prediction(torch.utils.data.Dataset):
    def __init__(self, name = 'cora', type = 'train'):
        self.G, self.node_feature, _, _, _, _ = load_data(name)
        self.node_feature = self.node_feature.toarray()
        self.node_feature = self.node_feature[:, np.newaxis, :]

        # G = max(nx.connected_component_subgraphs(G), key=len)
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.split_dataset()
        self.adj = nx.adjacency_matrix(self.G)
        self.adj_train = nx.adjacency_matrix(self.G_train)

    def split_dataset(self,thresh=0.2):
        self.G_train = self.G.copy()
        for edge in self.G_train.edges():
            self.G_train.remove_edge(edge[0],edge[1])
            if np.random.rand() > thresh or not nx.is_connected(self.G_train):
                self.G_train.add_edge(edge[0],edge[1])
        print('Train:', 'Connected', nx.is_connected(self.G_train),
              'Node', self.G_train.number_of_nodes(), 'Edge', self.G_train.number_of_edges())
        print('All:', 'Connected', nx.is_connected(self.G),
              'Node', self.G.number_of_nodes(), 'Edge', self.G.number_of_edges())
    def recompute_feature(self, G):
        # compute dist
        t1 = time.time()
        random_subsets = get_random_subsets(G, c=0.5)
        shortest_dists = nx.shortest_path_length(G)
        subset_dists, subset_ids = get_shortest_dists(shortest_dists, random_subsets, G.nodes())
        subset_features = get_feature(subset_ids, self.node_feature[:,0,:]) # remove extra dim

        subset_dists = subset_dists[:, :, np.newaxis]

        t2 = time.time()
        print('node num:', self.G.number_of_nodes(), 'subset num:', len(random_subsets),
              'time:', t2 - t1)
        return subset_dists, subset_features

    def __len__(self):
        return self.G_train.number_of_nodes()

    def __getitem__(self, idx): # todo: edit for link pred
        return self.node_feature[self.idx][idx], self.subset_dists[idx], self.subset_features[idx]

    def get_fullbatch_train(self):
        subset_dists, subset_features = self.recompute_feature(self.G_train)
        return (self.node_feature, self.adj_train.toarray(), subset_dists, subset_features)

    def get_fullbatch_test(self):
        subset_dists, subset_features = self.recompute_feature(self.G_train)
        return (self.node_feature, self.adj.toarray(), subset_dists, subset_features)