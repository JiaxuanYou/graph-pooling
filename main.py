import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time
from argparse import ArgumentParser



from dataloader import *
from encoders import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def make_args():
    parser = ArgumentParser()

    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=0, type=int)

    parser.add_argument('--output_dim', dest='output_dim', default=16, type=int)
    parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
    parser.add_argument('--num_epochs', dest='num_epochs', default=1000, type=int)
    parser.add_argument('--clip', dest='clip', default=2.0, type=float)

    return parser

args = make_args()

#### link prediction

# data loader
dataset_sampler_train = graph_dataset_link_prediction(type='train')

# model
if args.gpu:
    model = DeepBourgain(input_dim=dataset_sampler_train.node_feature.shape[2], output_dim=args.output_dim,
                         head_num=16, hidden_dim=16, has_out_act=False).cuda()
else:
    model = DeepBourgain(input_dim=dataset_sampler_train.node_feature.shape[2], output_dim=args.output_dim,
                     head_num=16, hidden_dim=16, has_out_act=False)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

out_act = nn.Sigmoid()
# train
for epoch in range(args.num_epochs):
    # train
    correct = 0
    total = 0

    model.train()
    model.zero_grad()

    batch = dataset_sampler_train.get_fullbatch_train()

    node_feature = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
    adj = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)
    subset_dists = Variable(torch.from_numpy(batch[2]).float(), requires_grad=False)
    subset_features = Variable(torch.from_numpy(batch[3]).float(), requires_grad=False)
    pred_raw = model(node_feature, subset_dists, subset_features)
    pred_before = pred_raw @ pred_raw.permute(1,0)
    # pred = out_act(pred_before)
    pred = pred_before

    loss = torch.mean((pred - adj)**2)
    loss.backward()
    # nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()

    # evaluate
    pred_binary = torch.where(pred > 0.5, torch.Tensor([1]), torch.Tensor([0]))
    correct += np.sum(adj.data.numpy() == pred_binary.data.numpy())
    total += adj.size()[0] * adj.size()[1]

    train_acc = correct/total
    # pdb.set_trace()


    # val
    correct = 0
    total = 0

    batch = dataset_sampler_train.get_fullbatch_test()

    node_feature = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
    adj = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)
    subset_dists = Variable(torch.from_numpy(batch[2]).float(), requires_grad=False)
    subset_features = Variable(torch.from_numpy(batch[3]).float(), requires_grad=False)
    pred = model(node_feature, subset_dists, subset_features)

    pred = out_act(pred @ pred.permute(1, 0))

    # evaluate
    pred_binary = torch.where(pred > 0.5, torch.Tensor([1]), torch.Tensor([0]))
    correct += np.sum(adj.data.numpy() == pred_binary.data.numpy())
    total += adj.size()[0] * adj.size()[1]

    val_acc = correct / total

    print('epoch', epoch, 'loss', loss.data, 'train accuracy', train_acc, 'val accuracy', val_acc)
    time.sleep(3)
        # if epoch==5:
        #     pdb.set_trace()








######## node classification

# dataset_sampler_train = graph_dataset_node_classification(type='train')
# dataset_sampler_val = graph_dataset_node_classification(type='val')
# dataset_sampler_test = graph_dataset_node_classification(type='test')
# dataset_loader_train = torch.utils.data.DataLoader(
#             dataset_sampler_train,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers)
# dataset_loader_val = torch.utils.data.DataLoader(
#             dataset_sampler_val,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers)
# dataset_loader_test = torch.utils.data.DataLoader(
#             dataset_sampler_test,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=args.num_workers)

# # model
# model = DeepBourgain(input_dim=dataset_sampler_train.node_feature.shape[2], output_dim=dataset_sampler_train.num_class, head_num=16, hidden_dim=16)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
#
# # train
# for epoch in range(args.num_epochs):
#     # train
#     correct = 0
#     total = 0
#     for i, batch in enumerate(dataset_loader_train):
#         model.train()
#         model.zero_grad()
#         node_feature = Variable(batch[0].float(), requires_grad=False)
#         node_label = Variable(batch[1].long(), requires_grad=False)
#         subset_dists = Variable(batch[2].float(), requires_grad=False)
#         subset_features = Variable(batch[3].float(), requires_grad=False)
#         pred = model(node_feature, subset_dists, subset_features)
#         loss = F.cross_entropy(pred, node_label, size_average=True)
#         loss.backward()
#         # nn.utils.clip_grad_norm(model.parameters(), args.clip)
#         optimizer.step()
#
#         # evaluate
#         pred_max = torch.argmax(pred.data, dim=-1).numpy()
#         correct += np.sum(node_label.data.numpy() == pred_max)
#         total += len(node_label)
#     train_acc = correct/total
#     dataset_sampler_train.recompute_feature()
#     dataset_loader_train = torch.utils.data.DataLoader(
#         dataset_sampler_train,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers)
#
#     # val
#     correct = 0
#     total = 0
#     for i, batch in enumerate(dataset_loader_val):
#         node_feature = Variable(batch[0].float(), requires_grad=False)
#         node_label = Variable(batch[1].long(), requires_grad=False)
#         subset_dists = Variable(batch[2].float(), requires_grad=False)
#         subset_features = Variable(batch[3].float(), requires_grad=False)
#         pred = model(node_feature, subset_dists, subset_features)
#
#         # evaluate
#         pred_max = torch.argmax(pred.data, dim=-1).numpy()
#         correct += np.sum(node_label.data.numpy() == pred_max)
#         total += len(node_label)
#     val_acc = correct/total
#     dataset_sampler_val.recompute_feature()
#     dataset_loader_val = torch.utils.data.DataLoader(
#         dataset_sampler_val,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers)
#
#     print('epoch', epoch, 'loss', loss.data, 'train accuracy', train_acc, 'val accuracy', val_acc)
#     time.sleep(3)
#         # if epoch==5:
#         #     pdb.set_trace()
#







def prepare_data():
    # load graph
    name = 'cora'
    plot = False
    # G = nx.grid_2d_graph(20,20)
    # G = nx.connected_caveman_graph(20,20)
    # G = nx.barabasi_albert_graph(1000,2)
    # G = nx.newman_watts_strogatz_graph(200,2,0.1)

    # deprecated
    # G, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    # G, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
    # G, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('pubmed')

    G, node_feature, label, idx_train, idx_val, idx_test = load_data('cora')
    node_feature = node_feature.toarray()
    node_label = np.zeros(label.shape[0])
    for i in range(label.shape[0]):
        node_label[i] = np.where(label[i]==1)[0][0]
    num_class = label.shape[-1]

    # G, features, labels = load_data('citeseer')
    # G, features, labels = load_data('pubmed')

    # G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)


    # compute dist
    t1 = time.time()
    random_subsets = get_random_subsets(G,c=0.5)
    shortest_dists = nx.shortest_path_length(G)
    subset_dists, subset_ids = get_shortest_dists(shortest_dists, random_subsets, G.nodes())
    subset_features = get_feature(subset_ids, node_feature)

    t2 = time.time()
    print('node num:', G.number_of_nodes())
    print('subset num:', len(random_subsets))
    print('time',t2-t1)




    if plot:
        node_emb = TSNE(n_components=2,n_iter=1000).fit_transform(subset_dists)
        print(node_emb.shape)

        # pca = PCA(n_components=2)
        # node_features_emb = pca.fit_transform(node_features)

        # plot results
        plt.figure()
        # nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, node_size=4, width=0.3, font_size = 3)
        nx.draw_networkx(G, pos=nx.spectral_layout(G), with_labels=True, node_size=4, width=0.3, font_size = 3)
        plt.savefig('fig/graph_'+name+str(len(random_subsets))+'.png')
        plt.close()
        # cmaps= ['b','g','r','c','m','y','k']
        # colors = []
        # for row in node_label:
        #     colors.append(np.where(row==1)[0][0])

        plt.figure()
        plt.scatter(node_emb[:, 0], node_emb[:, 1])

        # for i in range(node_features_emb.shape[0]):
        #     plt.scatter(node_emb[i,0],node_emb[i,1], c=cmaps[colors[i]],s=5)
        plt.savefig('fig/emb_'+name+str(len(random_subsets))+'.png')
        plt.close()

    node_feature = node_feature[:,np.newaxis,:]
    subset_dists = subset_dists[:,:,np.newaxis]
    subset_features = subset_features
    return node_feature, node_label, subset_dists, subset_features, num_class, idx_train, idx_val, idx_test

#
# node_feature, node_label, subset_dists, subset_features, num_class, idx_train, idx_val, idx_test = prepare_data()
# node_feature_train = node_feature[idx_train]
# node_feature_val = node_feature[idx_val]
# node_feature_test = node_feature[idx_test]
# node_label_train = node_label[idx_train]
# node_label_val = node_label[idx_val]
# node_label_test = node_label[idx_test]
# subset_dists_train = subset_dists[idx_train]
# subset_dists_val = subset_dists[idx_val]
# subset_dists_test = subset_dists[idx_test]
# subset_features_train = subset_features[idx_train]
# subset_features_val = subset_features[idx_val]



# subset_features_test = subset_features[idx_test]
#
#
# # svm on naive
# clf = SVC(gamma='auto')
# clf.fit(subset_dists_train[:,:,0], node_label_train)
# pred = clf.predict(subset_dists_test[:,:,0])
# correct = np.sum(node_label_test==pred)
# accuracy = correct/(len(node_label_test))
# print('svm accuracy', accuracy)
#
# clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0)
# clf.fit(subset_dists_train[:,:,0], node_label_train)
# pred = clf.predict(subset_dists_test[:,:,0])
# correct = np.sum(node_label_test==pred)
# accuracy = correct/(len(node_label_test))
# print('random forest accuracy', accuracy)


# node_feature_train = Variable(torch.from_numpy(node_feature_train).float(), requires_grad=False)
# node_feature_val = Variable(torch.from_numpy(node_feature_val).float(), requires_grad=False)
# node_feature_test = Variable(torch.from_numpy(node_feature_test).float(), requires_grad=False)
# node_label_train = Variable(torch.from_numpy(node_label_train).long(), requires_grad=False)
# node_label_val = Variable(torch.from_numpy(node_label_val).long(), requires_grad=False)
# node_label_test = Variable(torch.from_numpy(node_label_test).long(), requires_grad=False)
# subset_dists_train = Variable(torch.from_numpy(subset_dists_train).float(), requires_grad=False)
# subset_dists_val = Variable(torch.from_numpy(subset_dists_val).float(), requires_grad=False)
# subset_dists_test = Variable(torch.from_numpy(subset_dists_test).float(), requires_grad=False)
# subset_features_train = Variable(torch.from_numpy(subset_features_train).float(), requires_grad=False)
# subset_features_val = Variable(torch.from_numpy(subset_features_val).float(), requires_grad=False)
# subset_features_test = Variable(torch.from_numpy(subset_features_test).float(), requires_grad=False)
# # node_feature = Variable(torch.randn(128, 1, 128))
# # subset_dists = Variable(torch.randn(128, 64, 1))
# # subset_features = Variable(torch.randn(128, 64, 128))
# # pred = model(node_feature, subset_dists, subset_features)
#
#