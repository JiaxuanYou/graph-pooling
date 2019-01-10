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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report






def make_args():
    parser = ArgumentParser()

    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link; node')
    parser.add_argument('--model', dest='model', default='gcn', type=str,
                        help='deepbourgain; bourgain; gcn; gcnbourgain; node_feature; hybrid')
    parser.add_argument('--dataset', dest='dataset', default='grid', type=str,
                        help='grid; caveman; barabasi, cora, citeseer, pubmed')
    parser.add_argument('--loss', dest='loss', default='l2', type=str,
                        help='l2; cross_entropy')

    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')

    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int)
    parser.add_argument('--num_workers', dest='num_workers', default=0, type=int)

    parser.add_argument('--output_dim', dest='output_dim', default=16, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=16, type=int)
    parser.add_argument('--permute', dest='permute', action='store_true',
                        help='whether permute subsets')

    parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
    parser.add_argument('--num_epochs', dest='num_epochs', default=1000, type=int)
    parser.add_argument('--clip', dest='clip', default=2.0, type=float)

    parser.set_defaults(task='link', model='bourgain', dataset='grid',
                        gpu=False, permute=False)
    args = parser.parse_args()
    return args

args = make_args()
print(args)








def view_data():
    # load graph
    # G = nx.grid_2d_graph(20,20)
    # G = nx.connected_caveman_graph(20,20)
    # G = nx.barabasi_albert_graph(1000,2)
    # G = nx.newman_watts_strogatz_graph(200,2,0.1)

    dataset_sampler = graph_dataset_link_prediction(name=args.dataset, test_ratio = 0.02)
    G_raw = dataset_sampler.G_train_raw
    G = dataset_sampler.G_train
    print(G_raw.number_of_nodes(), G.number_of_nodes())

    subset_dists, _ = dataset_sampler.recompute_feature(G)
    subset_dists = np.squeeze(subset_dists)

    node_emb = TSNE(n_components=2,n_iter=1000).fit_transform(subset_dists)

    # pca = PCA(n_components=2)
    # node_emb = pca.fit_transform(subset_dists)

    print(node_emb.shape)

    # plot results
    plt.figure()
    plt.rcParams.update({'font.size': 4})
    # nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, node_size=4, width=0.3, font_size = 3)
    nx.draw_networkx(G_raw, pos=nx.spectral_layout(G_raw), with_labels=True, node_size=1.5, width=0.3, font_size = 4)
    plt.savefig('fig/graph_'+args.dataset+str(subset_dists.shape[1])+'.png', dpi=300)
    plt.close()
    # cmaps= ['b','g','r','c','m','y','k']
    # colors = []
    # for row in node_label:
    #     colors.append(np.where(row==1)[0][0])


    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 4})
    plt.scatter(node_emb[:, 0], node_emb[:, 1], s=1.5)

    for i, txt in enumerate(G_raw.nodes()):
        ax.annotate(txt, (node_emb[i, 0], node_emb[i, 1]))

    # for i in range(node_features_emb.shape[0]):
    #     plt.scatter(node_emb[i,0],node_emb[i,1], c=cmaps[colors[i]],s=5)
    plt.savefig('fig/emb_'+args.dataset+str(subset_dists.shape[1])+'.png', dpi=300)
    plt.close()


# view_data()

# quit()




#### link prediction
if args.task == 'link':
    # data loader
    dataset_sampler = graph_dataset_link_prediction(name=args.dataset)

    # model
    if args.model == 'deepbourgain':
        model = DeepBourgain(input_dim=dataset_sampler.node_feature.shape[2], output_dim=args.output_dim,
                             head_num=16, hidden_dim=16, has_out_act=False)
    elif args.model == 'gcn':
        model = GraphConv(input_dim=dataset_sampler.node_feature.shape[2], output_dim=args.output_dim,
                          normalize_embedding = True)
    elif args.model == 'bourgain':
        subset_dists, _ = dataset_sampler.recompute_feature(dataset_sampler.G_train)
        model = MLP(input_dim=subset_dists.shape[1], hidden_dim=16, output_dim=16)

    if args.gpu:
        model = model.cuda()



    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_act = nn.Sigmoid()

    # train
    for epoch in range(args.num_epochs):
        # train
        correct = 0
        total = 0

        model.train()
        model.zero_grad()

        batch = dataset_sampler.get_fullbatch_train()

        node_feature = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
        adj = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)
        subset_dists = Variable(torch.from_numpy(batch[2]).float(), requires_grad=False)
        subset_features = Variable(torch.from_numpy(batch[3]).float(), requires_grad=False)
        mask = Variable(torch.from_numpy(batch[4]).float(), requires_grad=False)
        if args.model == 'deepbourgain':
            pred = model(node_feature, subset_dists, subset_features)
        elif args.model == 'gcn':
            pred = model(node_feature, adj)
        elif args.model == 'bourgain':
            pred = model(subset_dists[:,:,0])
        adj_pred = pred @ pred.permute(1, 0)
        adj_pred = out_act(adj_pred)

        if args.loss == 'l2':
            loss = torch.mean((adj_pred - adj) ** 2 * mask) # todo cross entropy
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # evaluate
        adj_masked = torch.masked_select(adj, mask.byte())
        adj_pred_masked = torch.masked_select(adj_pred, mask.byte())

        adj_pred_masked_binary = torch.where(adj_pred_masked > 0.5, torch.Tensor([1]), torch.Tensor([0]))
        correct += np.sum(adj_masked.data.numpy() == adj_pred_masked_binary.data.numpy())
        total += adj_masked.size()[0]

        auc_train = roc_auc_score(adj_masked.flatten().data.numpy(), adj_pred_masked.flatten().data.numpy())

        acc_train = correct/total

        # print(classification_report(adj.flatten().data.numpy(), adj_pred_binary.flatten().data.numpy()))
        # pdb.set_trace()

        # val
        correct = 0
        total = 0

        batch = dataset_sampler.get_fullbatch_test()

        node_feature = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
        adj = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)
        subset_dists = Variable(torch.from_numpy(batch[2]).float(), requires_grad=False)
        subset_features = Variable(torch.from_numpy(batch[3]).float(), requires_grad=False)
        mask = Variable(torch.from_numpy(batch[4]).float(), requires_grad=False)


        if args.model == 'deepbourgain':
            pred = model(node_feature, subset_dists, subset_features)
        elif args.model == 'gcn':
            pred = model(node_feature, adj)
        elif args.model == 'bourgain':
            pred = model(subset_dists[:,:,0])
        adj_pred = pred @ pred.permute(1, 0)
        adj_pred = out_act(adj_pred)



        # evaluate
        adj_masked = torch.masked_select(adj, mask.byte())
        adj_pred_masked = torch.masked_select(adj_pred, mask.byte())

        adj_pred_masked_binary = torch.where(adj_pred_masked > 0.5, torch.Tensor([1]), torch.Tensor([0]))
        correct += np.sum(adj_masked.data.numpy() == adj_pred_masked_binary.data.numpy())
        total += adj_masked.size()[0]

        auc_test = roc_auc_score(adj_masked.flatten().data.numpy(), adj_pred_masked.flatten().data.numpy())

        acc_test = correct / total


        print('epoch', epoch, 'loss', loss.data, 'acc_train', acc_train, 'auc_train', auc_train,
            'acc_test', acc_test, 'auc_test', auc_test)
        time.sleep(3)
            # if epoch==5:
            #     pdb.set_trace()


######## node classification

elif args.task == 'node':
    # data loader
    dataset_sampler = graph_dataset_node_classification(name=args.dataset, permute=args.permute)

    # model
    if args.model == 'deepbourgain':
        model = DeepBourgain(input_dim=dataset_sampler.node_feature.shape[2],
                             output_dim=dataset_sampler.num_class, head_num=16, hidden_dim=args.hidden_dim, has_out_act=False)
    elif args.model == 'gcn':
        model = GraphConv(input_dim=dataset_sampler.node_feature.shape[2], output_dim=dataset_sampler.num_class,
                          normalize_embedding=True)

    elif args.model == 'gcnbourgain':
        # model = GraphConv_bourgain(input_dim=dataset_sampler.node_feature.shape[2], output_dim=dataset_sampler.num_class,
        #                            normalize_embedding=True, concat_bourgain=True)
        model = GCN_bourgain(input_dim=dataset_sampler.node_feature.shape[2], output_dim=dataset_sampler.num_class,
                             hidden_dim=args.hidden_dim, num_layers=2, concat=True, concat_bourgain=False)
    elif args.model == 'bourgain':
        subset_dists, _ = dataset_sampler.recompute_feature()
        model = MLP(input_dim=subset_dists.shape[1], hidden_dim=64, output_dim=dataset_sampler.num_class)

    elif args.model == 'node_feature':
        model = MLP(input_dim=dataset_sampler.node_feature.shape[2], hidden_dim=64, output_dim=dataset_sampler.num_class)

    elif args.model == 'hybrid':
        model_1 = DeepBourgain(input_dim=dataset_sampler.node_feature.shape[2],
                             output_dim=16, head_num=16, hidden_dim=16)
        model_2 = MLP(input_dim=dataset_sampler.node_feature.shape[2], hidden_dim=64, output_dim=16)

        model_combine = MLP(input_dim=16+16, hidden_dim=32, output_dim=dataset_sampler.num_class)

    if args.gpu:
        model = model.cuda()


    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.model == 'hybrid':
        optimizer = torch.optim.Adam(list(model_1.parameters())+list(model_2.parameters())+list(model_combine.parameters()),
                                     lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_act = nn.Sigmoid()

    # train
    for epoch in range(args.num_epochs):
        # train
        correct = 0
        total = 0

        if args.model == 'hybrid':
            model_1.zero_grad()
            model_2.zero_grad()
            model_combine.zero_grad()
        else:
            model.zero_grad()

        if 'gcn' in args.model:
            batch = dataset_sampler.get_fullbatch()
        else:
            batch = dataset_sampler.get_fullbatch_train()



        node_feature = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
        adj = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)
        node_label = Variable(torch.from_numpy(batch[2]).long(), requires_grad=False)
        subset_dists = Variable(torch.from_numpy(batch[3]).float(), requires_grad=False)
        subset_features = Variable(torch.from_numpy(batch[4]).float(), requires_grad=False)
        subset_ids = Variable(torch.from_numpy(batch[5]).long(), requires_grad=False)


        if args.model == 'deepbourgain':
            pred = model(node_feature, subset_dists, subset_features)
        elif args.model == 'gcn':
            pred = model(node_feature, adj)[dataset_sampler.idx_train]
            node_label = node_label[dataset_sampler.idx_train]
        elif args.model == 'gcnbourgain':
            pred = model(node_feature, adj, subset_dists, subset_ids)[dataset_sampler.idx_train]
            node_label = node_label[dataset_sampler.idx_train]
        elif args.model == 'bourgain':
            pred = model(subset_dists[:,:,0])
        elif args.model == 'node_feature':
            pred = model(node_feature[:,0,:])
        elif args.model == 'hybrid':
            pred_1 = model_1(node_feature, subset_dists, subset_features)
            pred_2 = model_2(node_feature[:,0,:])
            pred = model_combine(torch.cat((pred_1,pred_2),dim=-1))


        loss = F.cross_entropy(pred, node_label, size_average=True)
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # evaluate
        pred_max = torch.argmax(pred.data, dim=-1).numpy()
        correct += np.sum(node_label.data.numpy() == pred_max)
        total += len(node_label)

        train_acc = correct/total


        # val
        correct = 0
        total = 0


        if 'gcn' in args.model:
            batch = dataset_sampler.get_fullbatch()
        else:
            batch = dataset_sampler.get_fullbatch_val()

        node_feature = Variable(torch.from_numpy(batch[0]).float(), requires_grad=False)
        adj = Variable(torch.from_numpy(batch[1]).float(), requires_grad=False)
        node_label = Variable(torch.from_numpy(batch[2]).long(), requires_grad=False)
        subset_dists = Variable(torch.from_numpy(batch[3]).float(), requires_grad=False)
        subset_features = Variable(torch.from_numpy(batch[4]).float(), requires_grad=False)
        subset_ids = Variable(torch.from_numpy(batch[5]).long(), requires_grad=False)


        if args.model == 'deepbourgain':
            pred = model(node_feature, subset_dists, subset_features)
        elif args.model == 'gcn':
            pred = model(node_feature, adj)[dataset_sampler.idx_val]
            node_label = node_label[dataset_sampler.idx_val]
        elif args.model == 'gcnbourgain':
            pred = model(node_feature, adj, subset_dists, subset_ids)[dataset_sampler.idx_val]
            node_label = node_label[dataset_sampler.idx_val]
        elif args.model == 'bourgain':
            pred = model(subset_dists[:,:,0])
        elif args.model == 'node_feature':
            pred = model(node_feature[:,0,:])
        elif args.model == 'hybrid':
            pred_1 = model_1(node_feature, subset_dists, subset_features)
            pred_2 = model_2(node_feature[:,0,:])
            pred = model_combine(torch.cat((pred_1,pred_2),dim=-1))

        # evaluate
        pred_max = torch.argmax(pred.data, dim=-1).numpy()
        correct += np.sum(node_label.data.numpy() == pred_max)
        total += len(node_label)

        val_acc = correct/total

        # if epoch % 20 == 19:
        #     pdb.set_trace()

        print('epoch', epoch, 'loss', loss.data, 'train accuracy', train_acc, 'val accuracy', val_acc)

        # time.sleep(3) # lower consumption





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

# ######## node classification
#
# elif args.task == 'node':
#
# dataset_sampler_train = graph_dataset_node_classification(type='train')
# dataset_sampler_val = graph_dataset_node_classification(type='val')
# dataset_sampler_test = graph_dataset_node_classification(type='test')
# dataset_loader_train = torch.utils.data.DataLoader(
#     dataset_sampler_train,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=args.num_workers)
# dataset_loader_val = torch.utils.data.DataLoader(
#     dataset_sampler_val,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=args.num_workers)
# dataset_loader_test = torch.utils.data.DataLoader(
#     dataset_sampler_test,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=args.num_workers)
#
# # model
# model = DeepBourgain(input_dim=dataset_sampler_train.node_feature.shape[2], output_dim=dataset_sampler_train.num_class,
#                      head_num=16, hidden_dim=16)
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
#     train_acc = correct / total
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
#     val_acc = correct / total
#     dataset_sampler_val.recompute_feature()
#     dataset_loader_val = torch.utils.data.DataLoader(
#         dataset_sampler_val,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers)
#
#     print('epoch', epoch, 'loss', loss.data, 'train accuracy', train_acc, 'val accuracy', val_acc)
#     time.sleep(3)
#     # if epoch==5:
#     #     pdb.set_trace()