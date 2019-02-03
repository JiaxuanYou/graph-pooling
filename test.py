import networkx as nx
import pdb
from util import *
from encoders import *
from dataloader import *
# from main import make_args

import torch
#
# x = torch.ones(2,2,2)
#
# print(x.size())
# print(x.repeat(4,2,1).size())
#
# pdb.set_trace()




#
# # G = nx.DiGraph(nx.path_graph(4))
# G = nx.grid_2d_graph(50,50)
# G = nx.convert_node_labels_to_integers(G)
# personalization = {}
# for node in G.nodes():
#     personalization[node] = 0
# personalization[0]=1
# t1 = time.time()
# pr = nx.pagerank(G, alpha=0.9, personalization=personalization)
# t2 = time.time()
#
# pr_np = nx.pagerank_numpy(G, alpha=0.9, personalization=personalization)
#
# t3 = time.time()
# t4 = time.time()
# print(t2-t1, t3-t2, t4-t3)
#
# diff = 0
# for i in range(len(pr)):
#     diff += pr[i]-pr_np[i]
# print(diff)
#
# print(pr)
# print(pr_np)








# plt.figure()
# nx.draw(G)
# plt.savefig('fig/view.png')

# print([]+[1,2,3])
#
# G = nx.grid_2d_graph(5, 5)
# G = nx.convert_node_labels_to_integers(G)
# #
# #
# a = G.adjacency_list()
# pdb.set_trace()
# #
# adj = nx.adjacency_matrix(G)
#
# t1 = time.time()
# adj_2 = adj @ adj
# t2 = time.time()
#
# print(t2-t1)



# import random
#
# a = ['a', 'b', 'c']
# b = [1, 2, 3]
#
# c = list(zip(a, b))
#
# random.shuffle(c)
#
# a, b = zip(*c)
#
# print(list(a))
# print(b)


# a = np.zeros(1)
# b = np.zeros(1)
#
# c = (a,b)
#
# try:
#     c[0] += 100
# except:
#     pass
#
# print(c)
# print(c)
# print(c)
# print(c)
# print(c)






#
# G_raw =  nx.connected_caveman_graph(20, 20)
#
# nx.draw_networkx(G_raw, pos=nx.spring_layout(G_raw), with_labels=True, node_size=1.5, width=0.3,
#                      font_size=4)
#
# plt.savefig('fig/graph_test.png', dpi=300)
#
#
# import json
# import numpy as np
#
# from networkx.readwrite import json_graph
# from argparse import ArgumentParser
#
# dataset_dir = '/Users/jiaxuan/Downloads/ppi'
# print("Loading data...")
# G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
# labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
# labels = {int(i): l for i, l in labels.items()}
#
# train_ids = [n for n in G.nodes()]
# train_labels = np.array([labels[i] for i in train_ids])
# if train_labels.ndim == 1:
#     train_labels = np.expand_dims(train_labels, 1)
#
# print("Using only features..")
# feats = np.load(dataset_dir + "/ppi-feats.npy")
# ## Logistic gets thrown off by big counts, so log transform num comments and score
# feats[:, 0] = np.log(feats[:, 0] + 1.0)
# feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
# feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
# feat_id_map = {int(id): val for id, val in feat_id_map.items()}
# train_feats = feats[[feat_id_map[id] for id in train_ids]]
#
# # pdb.set_trace()
#
# components = nx.connected_components(G)
# components_len = [len(c) for c in components]
# print(components_len)

def make_args():
    parser = ArgumentParser()

    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link; node; linknew')
    parser.add_argument('--model', dest='model', default='gcn', type=str,
                        help='deepbourgain; bourgain; gcn; gcnbourgain; node_feature; hybrid; pgcn; gat')
    parser.add_argument('--dist_only', dest='dist_only', action='store_true',
                        help='whether dist_only')
    parser.add_argument('--dataset', dest='dataset', default='grid', type=str,
                        help='grid; caveman; barabasi, cora, citeseer, pubmed')
    parser.add_argument('--loss', dest='loss', default='l2', type=str,
                        help='l2; cross_entropy')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')

    # dataset
    parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.5, type=float)
    parser.add_argument('--graph_test_ratio', dest='graph_test_ratio', default=0.2, type=float)
    parser.add_argument('--permute', dest='permute', action='store_true',
                        help='whether permute subsets')
    parser.add_argument('--permute_no', dest='permute', action='store_false',
                        help='whether permute subsets')
    # parser.add_argument('--approximate', dest='approximate', action='store_true',
    #                     help='whether approximate dists')
    # parser.add_argument('--approximate_no', dest='approximate', action='store_false',
    #                     help='whether approximate dists')

    parser.add_argument('--approximate', dest='approximate', default=-1, type=int)

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=0, type=int)

    parser.add_argument('--num_layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--output_dim', dest='output_dim', default=16, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=16, type=int)

    parser.add_argument('--normalize_dist', dest='normalize_dist', action='store_true',
                        help='whether normalize_dist')
    parser.add_argument('--normalize_adj', dest='normalize_adj', action='store_true',
                        help='whether normalize_adj')


    parser.add_argument('--lr', dest='lr', default=1e-3, type=float)
    parser.add_argument('--num_epochs', dest='num_epochs', default=10, type=int)
    parser.add_argument('--num_repeats', dest='num_repeats', default=10, type=int)
    parser.add_argument('--clip', dest='clip', default=2.0, type=float)

    parser.set_defaults(gpu=False, task='linknew', model='pgcn', dataset='cora',
                        permute=True, approximate=-1, dist_only=False, normalize_adj=False)
    args = parser.parse_args()
    return args
args = make_args()
print(args)

dataset_sampler = graphs_dataset_loader_simple(name=args.dataset, remove_link_ratio=args.remove_link_ratio,
                                        graph_test_ratio=args.graph_test_ratio, permute=args.permute,
                                        approximate=args.approximate,
                                        normalize_adj=args.normalize_adj)

model = GCN(input_dim=dataset_sampler.graphs_feature[0].shape[1],
            hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers,
            normalize_embedding_l2 = True)


from sklearn.cluster import DBSCAN
# train
for epoch in range(1):
    # while True:
    correct = 0
    total = 0
    model.zero_grad()

    # if epoch==5:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] /= 10

    pred_all = []

    while True:
        batch = dataset_sampler.get_batch_train()

        adj = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
        feature = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)

        pred = model(feature, adj)
        pred_all.append(pred.data.numpy())

        if dataset_sampler.done_train:
            break

    X = np.concatenate(pred_all,axis=-1)
    # X = TSNE(n_components=2, n_iter=1000).fit_transform(X)
    # plt.scatter(X[:,0],X[:,1])
    # print(np.linalg.norm(pred_all,ord=2,axis=-1))

    db = DBSCAN(eps=0.2, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)

    plt.show()


