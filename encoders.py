import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import pdb
from torch.autograd import Variable


import numpy as np
from set2set import Set2Set

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act = nn.ReLU(), normalize_input=True):
        super(MLP, self).__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.act = act
        self.normalize_input = normalize_input

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
    def forward(self, x):
        # x = F.normalize(x, p=2, dim=-1)
        if self.normalize_input:
            x = (x-torch.mean(x,dim=0))/torch.std(x,dim=0)
        x = self.act(self.linear_1(x))
        return self.linear_2(x)


class DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act = nn.ReLU(), normalize_input=True, aggregation=torch.sum):
        super(DeepSet, self).__init__()

        self.mlp1 = MLP(input_dim=input_dim, hidden_dim=16, output_dim=1, act=nn.ReLU(), normalize_input=True)
        self.mlp2 = MLP(input_dim=input_dim, hidden_dim=16, output_dim=1, act=nn.ReLU(), normalize_input=True)
        self.agg = aggregation

    def forward(self, x):
        # dim ?*?*d
        x = self.mlp1(x)
        x = self.agg(x,dim=-2)



class MultiAttention(nn.Module):
    def __init__(self, input_dim, head_num):
        super(MultiAttention, self).__init__()
        # todo: vectorize
        self.head_num = head_num
        self.weights = [nn.Parameter(torch.FloatTensor(input_dim, input_dim)) for i in range(head_num)]

        for weight in self.weights:
            weight.data = init.xavier_uniform_(weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x1, x2):
        pred = [0 for i in range(self.head_num)]
        for i in range(self.head_num):
            pred[i] = x1 @ self.weights[i] @ x2.permute(0, 2, 1)  # return n*m*1
        pred = torch.cat(pred, dim=2)  # n*m*head
        return pred


class DeepBourgain(nn.Module):
    def __init__(self, input_dim, output_dim, head_num, hidden_dim,
                 has_out_act = True, out_act = nn.Softmax(dim=-1), func_type='gcn',
                 normalize_embedding = False):
        '''

        :param input_dim: node dim d
        :param output_dim:
        :param hidden_dim:
        :param head_num: number of attention heads

        '''
        # todo: compatiable with cuda
        super(DeepBourgain, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.feature_1 = nn.Linear(input_dim, hidden_dim)
        self.feature_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention = MultiAttention(hidden_dim, head_num)
        self.deepset_1 = nn.Linear(head_num+1, hidden_dim)
        self.deepset_2 = nn.Linear(hidden_dim, hidden_dim)

        self.agg_1 = nn.Linear(input_dim*2, hidden_dim)
        self.agg_2 = nn.Linear(hidden_dim, hidden_dim)

        self.dist_1 = nn.Linear(hidden_dim+1,hidden_dim)
        self.dist_2 = nn.Linear(hidden_dim,hidden_dim)

        self.out_1 = nn.Linear(hidden_dim, output_dim)
        self.out_2 = nn.Linear(input_dim, output_dim)

        self.dist_compute = MLP(input_dim=1, hidden_dim=16, output_dim=1, act=nn.ReLU(), normalize_input=False)

        self.feature_compute = MLP(input_dim=input_dim, hidden_dim=16, output_dim=16, act=nn.ReLU(), normalize_input=False)

        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.has_out_act = has_out_act
        self.out_act = out_act

        self.func_type = func_type
        self.normalize_embedding = normalize_embedding
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    # def forward(self, node_feature, subset_dists, subset_features):
    #     if self.func_type == 'deepset':
    #         return self.forward_deepset(node_feature, subset_dists, subset_features)
    #     elif self.func_type == 'gcn':
    #         return self.forward_gcn(node_feature, subset_dists, subset_features)



    def forward_deepset(self, node_feature, subset_dists, subset_features):
        '''
        n: nodes, m: subsets, d: dim of node feature, h: dim of hidden
        attention style

        :param node_feature: n*1*d
        :param subset_dists: n*m*1
        :param subset_features: n*m*d
        :return:
        '''
        # 1 embed node/subset feature
        subset_features = self.act(self.feature_1(subset_features))
        subset_features = self.feature_2(subset_features) # n*m*h
        node_feature = self.act(self.feature_1(node_feature))
        node_feature = self.feature_2(node_feature) # n*1*h

        # 2 multi-head attention, bilinear style
        pred = self.attention(subset_features, node_feature)

        # 3 DeepSet aggregation
        pred = self.act(self.deepset_1(torch.cat((pred,subset_dists),dim=-1)))
        pred = self.deepset_2(pred) # n*m*out
        # weighted by the distance, todo: better way to use distance information
        # pred = pred / (subset_dists+1)
        pred = torch.mean(pred, dim=1) # n*out

        # 4 output
        pred = self.out_1(pred) # n*out
        # pred = F.normalize(pred, p=2, dim=-1)
        if self.has_out_act:
            pred = self.out_act(pred)

        return pred

    # def forward_gcn(self, node_feature, subset_dists, subset_features):
    #     '''
    #     n: nodes, m: subsets, d: dim of node feature, h: dim of hidden
    #     attention style
    #
    #     :param node_feature: n*1*d
    #     :param subset_dists: n*m*1
    #     :param subset_features: n*m*d
    #     :return:
    #     '''
    #
    #     # 1 concat node feature with subset features
    #     node_feature = torch.cat((node_feature.repeat(1,subset_dists.size()[1],1),subset_features),dim = 2)
    #     node_feature = self.act(self.agg_1(node_feature))
    #     pred = self.agg_2(node_feature)
    #
    #     subset_dists = self.dist_compute(subset_dists)
    #
    #     # subset_dists = 1 / (subset_dists + 1)
    #
    #     # option 1: weighted by dist then sum
    #     # subset_dists = self.dist_1(subset_dists)
    #     # print(self.dist_1.weight, self.dist_1.bias)
    #     pred = pred * subset_dists
    #
    #     # option 2: concat with dist
    #     # pred = self.act(self.dist_1(torch.cat((pred, subset_dists), dim=-1)))
    #     # pred = self.dist_2(pred)  # n*m*out
    #
    #     pred = torch.mean(pred, dim=1) # n*out
    #
    #     # 4 output
    #     pred = self.out_1(pred) # n*out
    #     if self.normalize_embedding:
    #         pred = F.normalize(pred, p=2, dim=-1)
    #     if self.has_out_act:
    #         pred = self.out_act(pred)
    #     pdb.set_trace()
    #     return pred


    def forward_gcn(self, node_feature, subset_dists, subset_features):
        '''
        n: nodes, m: subsets, d: dim of node feature, h: dim of hidden
        attention style

        :param node_feature: n*1*d
        :param subset_dists: n*m*1
        :param subset_features: n*m*d
        :return:
        '''

        # 1 concat node feature with subset features

        subset_dists = self.dist_compute(subset_dists)
        subset_features = self.feature_compute(subset_features)
        pred = subset_dists * subset_features
        pred = torch.mean(pred, dim=1)


        # option 2: concat with dist
        # pred = self.act(self.dist_1(torch.cat((pred, subset_dists), dim=-1)))
        # pred = self.dist_2(pred)  # n*m*out

        # pred = torch.mean(pred, dim=1) # n*out

        # 4 output
        # pred = self.out_1(pred) # n*out
        # if self.normalize_embedding:
        # pred = F.normalize(pred, p=2, dim=-1)
        # if self.has_out_act:
        #     pred = self.out_act(pred)
        # pdb.set_trace()
        return pred
        # return torch.squeeze(subset_dists)

    # def forward_gcn_approximate(self, node_feature, adj_lists_concat):
    def forward(self, node_feature, adj_count):
        '''
        n: nodes, m: subsets, d: dim of node feature, h: dim of hidden
        attention style

        :param node_feature: n*1*d
        :param subset_dists: n*m*1
        :param subset_features: n*m*d
        :return:
        '''

        # 1 concat node feature with subset features

        # subset_dists = self.dist_compute(subset_dists)

        # adj_count = self.softmax(adj_count)
        # adj_count = adj_count - torch.mean(adj_count, dim=0)

        # node_feature = self.feature_compute(node_feature[:,0,:])
        node_feature = node_feature[:,0,:]


        adj_count = adj_count / torch.sum(adj_count,dim=-1,keepdim=True)
        pred = adj_count @ node_feature
        # pred = pred/torch.sum(adj_count,dim=-1,keepdim=True)

        # pred = pred + node_feature
        pred = self.out_2(pred)

        pred = F.normalize(pred, p=2, dim=-1)
        # pred = torch.cat((pred, node_feature), dim=-1)
        # pred = self.out_2(pred)


        # pdb.set_trace()

        # pdb.set_trace()
        # pdb.set_trace()

        # option 2: concat with dist
        # pred = self.act(self.dist_1(torch.cat((pred, subset_dists), dim=-1)))
        # pred = self.dist_2(pred)  # n*m*out

        # pred = torch.mean(pred, dim=1) # n*out

        # 4 output
        # pred = self.out_1(pred) # n*out
        # if self.normalize_embedding:
        # pred = F.normalize(pred, p=2, dim=-1)
        # if self.has_out_act:
        #     pred = self.out_act(pred)
        # pdb.set_trace()
        return pred
        # return torch.squeeze(subset_dists)



# # # GCN basic operation
# class GraphConv(nn.Module):
#     def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
#                  dropout=0.0, bias=True):
#         super(GraphConv, self).__init__()
#         self.add_self = add_self
#         self.dropout = dropout
#         if dropout > 0.001:
#             self.dropout_layer = nn.Dropout(p=dropout)
#         self.normalize_embedding = normalize_embedding
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
#         self.weight.data = init.xavier_uniform(self.weight.data, gain=nn.init.calculate_gain('relu'))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(output_dim))
#             self.bias.data = init.constant(self.bias.data, 0.0)
#         else:
#             self.bias = None
#
#     def forward(self, x, adj):
#         x = x.squeeze(1)
#         if self.dropout > 0.001:
#             x = self.dropout_layer(x)
#         y = torch.matmul(adj, x)
#         if self.add_self:
#             y += x
#         y = torch.matmul(y, self.weight)
#         if self.bias is not None:
#             y = y + self.bias
#         if self.normalize_embedding:
#             y = F.normalize(y, p=2, dim=-1)
#             # print(y[0][0])
#         return y





# # # GCN basic operation
# class GraphConv(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, normalize_embedding=True):
#         super(GraphConv, self).__init__()
#         self.normalize_embedding = normalize_embedding
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         self.out_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
#                                output_dim=output_dim, act=nn.ReLU(), normalize_input=False)
#
#
#     def forward(self, x, adj):
#         pred = torch.matmul(adj, x)
#
#         pred = self.out_compute(pred)
#         if self.normalize_embedding:
#             # pred = F.normalize(pred, p=2, dim=-1)
#             pred = (pred - torch.mean(pred, dim=0)) / torch.std(pred, dim=0)
#         return pred

# # GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, normalize_embedding=False,
                 normalize_embedding_l2=False, att=False, mpnn=False, graphsage=False):
        super(GraphConv, self).__init__()
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_l2 = normalize_embedding_l2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.att = att
        self.mpnn = mpnn
        self.graphsage = graphsage

        if self.graphsage:
            self.out_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                   output_dim=output_dim, act=nn.ReLU(), normalize_input=False)
        elif self.mpnn:
            self.out_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                   output_dim=output_dim, act=nn.ReLU(), normalize_input=False)
        else:
            self.out_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                               output_dim=output_dim, act=nn.ReLU(), normalize_input=False)
        if self.att:
            self.att_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                   output_dim=output_dim, act=nn.LeakyReLU(0.2), normalize_input=False)
        if self.mpnn:
            self.mpnn_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                   output_dim=hidden_dim, act=nn.ReLU(), normalize_input=False)

        # self.W = nn.Parameter(torch.zeros(size=(input_dim, input_dim)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, adj):
        if self.att:
            x_att = self.att_compute(x)
            # pdb.set_trace()
            att = x_att @ x_att.permute(1,0)
            # pdb.set_trace()
            att = self.softmax(att)
            # pdb.set_trace()
            pred = torch.matmul(adj*att, x)
            # pdb.set_trace()
        elif self.mpnn:
            x1 = x.unsqueeze(0).repeat(x.shape[0],1,1)
            # x2 = x1.permute(1,0,2)
            x2 = x.unsqueeze(1).repeat(1,x.shape[0],1)
            e = torch.cat((x1,x2),dim=-1)
            e = self.mpnn_compute(e)
            pred = torch.mean(adj.unsqueeze(-1)*e, dim=1)
            # return pred
        else:
            pred = torch.matmul(adj, x)
        # pdb.set_trace()
        if self.graphsage:
            pred = torch.cat((pred,x),dim=-1)

        pred = self.out_compute(pred)
        # pdb.set_trace()
        if self.normalize_embedding:
            pred = (pred - torch.mean(pred, dim=0)) / torch.std(pred, dim=0)
        if self.normalize_embedding_l2:
            pred = F.normalize(pred, p=2, dim=-1)
        # pdb.set_trace()
        return pred



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -10 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime



# # # Position GCN basic operation
# class PositionGraphConv(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, normalize_embedding=True,aggregation=torch.sum,dist_only=False, approximate=True):
#         super(PositionGraphConv, self).__init__()
#         self.normalize_embedding = normalize_embedding
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.aggregation = aggregation
#         self.dist_only = dist_only
#         self.approximate  = approximate
#
#
#         self.feature_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
#                                    output_dim=hidden_dim, act=nn.ReLU(), normalize_input=False)
#         self.out_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
#                                output_dim=1, act=nn.ReLU(), normalize_input=True)
#         self.hidden_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
#                                output_dim=output_dim, act=nn.ReLU(), normalize_input=False)
#
#
#         # self.out_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
#         #                        output_dim=1, act=nn.ReLU(), normalize_input=True)
#         # self.hidden_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
#         #                           output_dim=output_dim, act=nn.ReLU(), normalize_input=True)
#
#     def forward(self, feature, dist, anchors, output_hidden=False):
#         n = feature.shape[0]
#         messages = []
#         for anchor in anchors:
#             if self.dist_only:
#                 messages.append(torch.min(dist[:,anchor], dim=-1, keepdim=True)[0])
#                 continue
#             # select feature
#             self_temp = feature.unsqueeze(1).repeat(1,len(anchor),1)
#             feature_temp = feature[anchor,:].unsqueeze(0).repeat(n,1,1)
#             feature_temp = torch.cat((feature_temp,self_temp),dim=-1)
#             dist_temp = dist[:,anchor].unsqueeze(2)
#
#             # weighted sum
#             message = feature_temp * dist_temp
#             # pdb.set_trace()
#
#             message = self.feature_compute(message)
#             # pdb.set_trace()
#
#             # if self.approximate:
#             message = torch.sum(message,dim=1,keepdim=True)
#             # else:
#             #     message = torch.sum(message,dim=1,keepdim=True)/len(anchor)
#             # if self.normalize_embedding:
#             #     pred_temp = F.normalize(pred_temp, p=2, dim=-1)
#             messages.append(message)
#             # pdb.set_trace()
#
#         messages = torch.cat(messages,dim=1)
#         if self.dist_only:
#             out = messages
#         else:
#             out = self.out_compute(messages).squeeze()
#         # pdb.set_trace()
#         if self.normalize_embedding:
#             out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
#             # out = F.normalize(out, p=2, dim=-1)
#         if not output_hidden:
#             return out
#
#         hidden = self.hidden_compute(messages)
#         hidden = torch.mean(hidden,dim=1)
#         if self.normalize_embedding:
#             hidden = (hidden - torch.mean(hidden, dim=0)) / torch.std(hidden, dim=0)
#         return out, hidden







# # Position GCN basic operation
class PositionGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, normalize_embedding=True,aggregation=torch.sum,dist_only=False,concat=True,feature_pre=True, normalize_input=False):
        super(PositionGraphConv, self).__init__()
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.dist_only = dist_only
        self.concat = concat
        self.feature_pre = feature_pre
        self.normalize_input = normalize_input


        # self.feature_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
        #                            output_dim=hidden_dim, act=nn.ReLU(), normalize_input=False)
        # self.out_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                        output_dim=1, act=nn.ReLU(), normalize_input=False)
        # self.hidden_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                        output_dim=output_dim, act=nn.ReLU(), normalize_input=False)



        self.dist_compute = MLP(input_dim=1, hidden_dim=hidden_dim,
                               output_dim=1, act=nn.ReLU(), normalize_input=normalize_input)
        if self.concat:
            self.out_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                   output_dim=1, act=nn.ReLU(), normalize_input=normalize_input)
            self.hidden_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                      output_dim=output_dim, act=nn.ReLU(), normalize_input=normalize_input)
        else:
            self.out_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                   output_dim=1, act=nn.ReLU(), normalize_input=normalize_input)
            self.hidden_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                      output_dim=output_dim, act=nn.ReLU(), normalize_input=normalize_input)
        # if self.feature_pre:
        #     self.feature_pre = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
        #                            output_dim=hidden_dim, act=nn.ReLU(), normalize_input=)


    def forward(self, feature, dist, dist_max, dist_argmax, output_hidden=False):
        if self.dist_only:
            out = self.dist_compute(dist_max.unsqueeze(-1)).squeeze()
            if self.normalize_embedding:
                out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
            return out




        subset_features = feature[dist_argmax.flatten(), :]
        subset_features = subset_features.reshape((dist_argmax.shape[0], dist_argmax.shape[1],
                                                   feature.shape[1]))
        # pdb.set_trace()
        if self.concat:
            self_feature = feature.unsqueeze(1).repeat(1, dist_max.shape[1], 1)
            subset_features = torch.cat((subset_features, self_feature), dim=-1)
        messages = subset_features * dist_max.unsqueeze(-1)
        # pdb.set_trace()
        out = self.out_compute(messages).squeeze()
        # pdb.set_trace()
        # pdb.set_trace()
        if self.normalize_embedding:
            out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
            # out = F.normalize(out, p=2, dim=-1)
        # pdb.set_trace()
        if not output_hidden:
            return out

        hidden = self.hidden_compute(messages)
        hidden = torch.mean(hidden,dim=1)
        if self.normalize_embedding:
            hidden = (hidden - torch.mean(hidden, dim=0)) / torch.std(hidden, dim=0)
        return out, hidden





# # Ordered GCN basic operation, where neighbourhood has canonical ordering
class OrderedGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, normalize_embedding=True,aggregation=torch.sum,dist_only=False,concat=True,feature_pre=True, normalize_input=False):
        super(PositionGraphConv, self).__init__()
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.dist_only = dist_only
        self.concat = concat
        self.feature_pre = feature_pre
        self.normalize_input = normalize_input


        # self.feature_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
        #                            output_dim=hidden_dim, act=nn.ReLU(), normalize_input=False)
        # self.out_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                        output_dim=1, act=nn.ReLU(), normalize_input=False)
        # self.hidden_compute = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                        output_dim=output_dim, act=nn.ReLU(), normalize_input=False)



        self.dist_compute = MLP(input_dim=1, hidden_dim=hidden_dim,
                               output_dim=1, act=nn.ReLU(), normalize_input=normalize_input)
        if self.concat:
            self.out_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                   output_dim=1, act=nn.ReLU(), normalize_input=normalize_input)
            self.hidden_compute = MLP(input_dim=input_dim*2, hidden_dim=hidden_dim,
                                      output_dim=output_dim, act=nn.ReLU(), normalize_input=normalize_input)
        else:
            self.out_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                   output_dim=1, act=nn.ReLU(), normalize_input=normalize_input)
            self.hidden_compute = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                                      output_dim=output_dim, act=nn.ReLU(), normalize_input=normalize_input)
        # if self.feature_pre:
        #     self.feature_pre = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
        #                            output_dim=hidden_dim, act=nn.ReLU(), normalize_input=)


    def forward(self, feature, dist, dist_max, dist_argmax, output_hidden=False):
        if self.dist_only:
            out = self.dist_compute(dist_max.unsqueeze(-1)).squeeze()
            if self.normalize_embedding:
                out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
            return out




        subset_features = feature[dist_argmax.flatten(), :]
        subset_features = subset_features.reshape((dist_argmax.shape[0], dist_argmax.shape[1],
                                                   feature.shape[1]))
        # pdb.set_trace()
        if self.concat:
            self_feature = feature.unsqueeze(1).repeat(1, dist_max.shape[1], 1)
            subset_features = torch.cat((subset_features, self_feature), dim=-1)
        messages = subset_features * dist_max.unsqueeze(-1)
        # pdb.set_trace()
        out = self.out_compute(messages).squeeze()
        # pdb.set_trace()
        # pdb.set_trace()
        if self.normalize_embedding:
            out = (out - torch.mean(out, dim=0)) / torch.std(out, dim=0)
            # out = F.normalize(out, p=2, dim=-1)
        # pdb.set_trace()
        if not output_hidden:
            return out

        hidden = self.hidden_compute(messages)
        hidden = torch.mean(hidden,dim=1)
        if self.normalize_embedding:
            hidden = (hidden - torch.mean(hidden, dim=0)) / torch.std(hidden, dim=0)
        return out, hidden
















class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers = 2, concat=False,
                 normalize_embedding=True,normalize_embedding_l2=False, att=False, mpnn=False, graphsage=False):
        super(GCN, self).__init__()
        self.concat = concat
        self.att = att
        self.num_layers = num_layers
        self.conv_first = GraphConv(input_dim=input_dim, hidden_dim=hidden_dim,
                                    output_dim=hidden_dim, normalize_embedding=normalize_embedding, normalize_embedding_l2=normalize_embedding_l2,
                                    att=att, mpnn=mpnn,graphsage=graphsage)

        if self.num_layers > 1:
            self.conv_block = nn.ModuleList([GraphConv(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                       output_dim=hidden_dim, normalize_embedding=normalize_embedding,
                                                       normalize_embedding_l2=normalize_embedding_l2, att=att, mpnn=mpnn,graphsage=graphsage)
                                             for i in range(num_layers - 2)])

            self.conv_last = GraphConv(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                       output_dim=hidden_dim, normalize_embedding=normalize_embedding, normalize_embedding_l2=normalize_embedding_l2,
                                       att=att, mpnn=mpnn,graphsage=graphsage)
        if self.concat:
            self.MLP = MLP(input_dim=hidden_dim*num_layers, hidden_dim=hidden_dim,
                           output_dim=output_dim, act=nn.ReLU(), normalize_input=True)
        else:
            self.MLP = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
                           output_dim=output_dim, act=nn.ReLU(), normalize_input=True)
        self.act = nn.ReLU()
        self.w = nn.Parameter(torch.zeros([1]))
        self.w.data = nn.init.constant_(self.w, 1)
        self.b = nn.Parameter(torch.zeros([1]))
        self.b.data = nn.init.constant_(self.b, 0)

    def forward(self, x, adj):
        x = self.conv_first(x, adj)
        x = self.act(x)
        # x_all = [x]
        if self.num_layers>1:
            for i in range(len(self.conv_block)):
                x = self.conv_block[i](x, adj)
                x = self.act(x)
                # x_all.append(x)
            x = self.conv_last(x, adj)
            # x_all.append(x)
        # if self.concat:
        #     x = torch.cat(x_all, dim = 1)
        # x = self.MLP(x)

        return x





class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.w = nn.Parameter(torch.zeros([1]))
        self.w.data = nn.init.constant_(self.w, 1)
        self.b = nn.Parameter(torch.zeros([1]))
        self.b.data = nn.init.constant_(self.b, 0)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        # x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)
        return x








class PGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers = 2,
                 normalize_embedding=True,aggregation=torch.sum,dist_only=False):
        super(PGNN, self).__init__()
        self.num_layers = num_layers
        self.conv_first = PositionGraphConv(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=hidden_dim, normalize_embedding=normalize_embedding,aggregation=aggregation,dist_only=dist_only)

        if self.num_layers > 1:
            self.conv_block = nn.ModuleList([PositionGraphConv(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                       output_dim=hidden_dim, normalize_embedding=normalize_embedding,
                                                               aggregation=aggregation,dist_only=dist_only)
                                             for i in range(num_layers - 2)])

            self.conv_last = PositionGraphConv(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                       output_dim=output_dim, normalize_embedding=normalize_embedding,
                                               aggregation=aggregation,dist_only=dist_only)

        self.act = nn.ReLU()
        self.w = nn.Parameter(torch.zeros([1]))
        self.w.data = nn.init.constant_(self.w, 1)
        self.b = nn.Parameter(torch.zeros([1]))
        self.b.data = nn.init.constant_(self.b, 0)
    def forward(self, feature, dist, dist_max, dist_argmax):
        if self.num_layers>1:
            _, feature = self.conv_first(feature, dist, dist_max, dist_argmax, output_hidden=True)
            for i in range(len(self.conv_block)):
                _, feature = self.conv_block[i](feature, dist, dist_max, dist_argmax, output_hidden=True)
            pred = self.conv_last(feature, dist, dist_max, dist_argmax)
        else:
            pred = self.conv_first(feature, dist, dist_max, dist_argmax)

        return pred


















class GraphConv_bourgain(nn.Module):
    def __init__(self, input_dim, output_dim, concat_bourgain=False, concat_self=False,
                 normalize_embedding=False, dropout=0.0, bias=True):
        super(GraphConv_bourgain, self).__init__()
        self.concat_bourgain = concat_bourgain
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim*(int(concat_self)+int(concat_bourgain)+1),
                                                     output_dim))
        self.weight.data = init.xavier_uniform(self.weight.data, gain=nn.init.calculate_gain('relu'))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            self.bias.data = init.constant(self.bias.data, 0.0)
        else:
            self.bias = None

    def forward(self, node_feature_raw, adj, subset_dists, subset_ids):
        node_feature = node_feature_raw.squeeze(1)
        if self.dropout > 0.001:
            node_feature = self.dropout_layer(node_feature)
        y = torch.matmul(adj, node_feature)
        if self.concat_bourgain:
            subset_features = node_feature[subset_ids.flatten(), :]
            subset_features = subset_features.reshape((subset_ids.shape[0], subset_ids.shape[1],
                                                       node_feature.shape[1]))
            subset_features = subset_features / (subset_dists+1)
            subset_features = torch.mean(subset_features, dim=1)  # n*out
            y = torch.cat((subset_features,y), dim=-1)
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=-1)
            # print(y[0][0])
        return y




class GCN_bourgain(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers = 2, concat=False, concat_bourgain=False,
                 normalize_embedding=False, dropout=0.0, bias=True):
        super(GCN_bourgain, self).__init__()
        self.concat = concat
        self.conv_first = GraphConv_bourgain(input_dim=input_dim, output_dim=hidden_dim, concat_bourgain=concat_bourgain,
                               normalize_embedding=normalize_embedding, bias=bias)

        self.conv_block = nn.ModuleList([GraphConv_bourgain(input_dim=hidden_dim, output_dim=hidden_dim,
                concat_bourgain=concat_bourgain, normalize_embedding=normalize_embedding, bias=bias)
                                         for i in range(num_layers - 2)])

        self.conv_last = GraphConv_bourgain(input_dim=hidden_dim, output_dim=hidden_dim,
                                             concat_bourgain=concat_bourgain,
                                             normalize_embedding=normalize_embedding, bias=bias)
        if self.concat:
            self.MLP = nn.Sequential(nn.Linear(hidden_dim*num_layers, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))
        else:
            self.MLP = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, output_dim))
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, GraphConv_bourgain):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, subset_dists, subset_ids):
        x = self.conv_first(x, adj, subset_dists, subset_ids)
        x = self.act(x)
        x_all = [x]
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x, adj, subset_dists, subset_ids)
            x = self.act(x)
            x_all.append(x)
        x = self.conv_last(x, adj, subset_dists, subset_ids)
        x_all.append(x)
        if self.concat:
            x = torch.cat(x_all, dim = 1)
        x = self.MLP(x)
        return x



#
# # GCN basic operation
# class GraphConv(nn.Module):
#     def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
#             dropout=0.0, bias=True):
#         super(GraphConv, self).__init__()
#         self.add_self = add_self
#         self.dropout = dropout
#         if dropout > 0.001:
#             self.dropout_layer = nn.Dropout(p=dropout)
#         self.normalize_embedding = normalize_embedding
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
#         else:
#             self.bias = None
#
#     def forward(self, x, adj):
#         if self.dropout > 0.001:
#             x = self.dropout_layer(x)
#         y = torch.matmul(adj, x)
#         if self.add_self:
#             y += x
#         y = torch.matmul(y,self.weight)
#         if self.bias is not None:
#             y = y + self.bias
#         if self.normalize_embedding:
#             y = F.normalize(y, p=2, dim=2)
#             #print(y[0][0])
#         return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        #print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
            
        #return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        #out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = []
        self.assign_conv_block_modules = []
        self.assign_conv_last_modules = []
        self.assign_pred_modules = []
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            self.assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)


            # next pooling layer
            assign_input_dim = embedding_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)
            self.assign_pred_modules.append(self.assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        #self.assign_tensor = self.gcn_forward(x_a, adj, 
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        #self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        #if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
        
            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])


            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)


        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1-adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss

