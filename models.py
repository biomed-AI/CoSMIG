
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import MessagePassing, GCNConv, RGCNConv, global_sort_pool, global_add_pool
from torch_geometric.utils import dropout_adj
from dataset import *
import pdb
import time

from typing import Optional, Union, Tuple

class GNN(torch.nn.Module):
    # a base GNN class, GCN message passing + sum_pooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], 
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(GNN, self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout 
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
        self.lin1 = Linear(sum(latent_dim), 128)
        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_add_pool(concat_states, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class DGCNN(GNN):
    # DGCNN from [Zhang et al. AAAI 2018], GCN message passing + SortPooling
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[32, 32, 32, 1], k=30, 
                 regression=False, adj_dropout=0.2, force_undirected=False):
        super(DGCNN, self).__init__(
            dataset, gconv, latent_dim, regression, adj_dropout, force_undirected
        )
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)
        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        x = global_sort_pool(concat_states, batch, self.k)  # batch * (k*hidden)
        x = x.unsqueeze(1)  # batch * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # flatten
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0]
        else:
            return F.log_softmax(x, dim=-1)


class IGMC(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):
        super(IGMC, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2*sum(latent_dim)+n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)


from layer import Communicate_GCNConv, BatchGRU
class CoSMIG(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use Communicative GNN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=Communicate_GCNConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, num_bases=2, regression=False, adj_dropout=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):
        super(CoSMIG, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by
        self.num_relations = num_relations
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.gru = BatchGRU(sum(latent_dim))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2*sum(latent_dim)+n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout, 
                force_undirected=self.force_undirected, num_nodes=len(x), 
                training=self.training
            )
        edge_attr = F.one_hot(edge_type, self.num_relations).to(torch.float)
        edge_attr = edge_attr.to(device=x.device)

        concat_states = []
        for conv in self.convs:
            x, edge_attr = conv(x, edge_index, edge_type, edge_attr)
            x = torch.tanh(x)
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        # concat_states = self.gru(concat_states, batch)

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)



# class CoSMIG(GNN):
#     r"""
#     A CoSMIG is a model which contains a message passing network 
#     following by feed-forward layers.
#     """

#     def __init__(self, dataset,
#                    hidden_channels: int = 256,
#                    bias: bool = False,
#                    depth: int = 3,
#                    dropout: float = 0.0,
#                    undirected: bool = False,
#                    multiply_by: int = 1,
#                    regression: bool = False,
#                    side_features: bool = False,
#                    n_side_features: int = 0, 
#                    num_relations: int = 5, 
#                    num_bases: int = 2):
#         """
#         Initializes the CMPNN.

#         :param classification: Whether the model is a classification model.
#         """
#         latent_dim = [hidden_channels] * 4
#         adj_dropout = dropout
#         force_undirected = True

#         super(CoSMIG, self).__init__(
#             dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
#         )
#         self.multiply_by = multiply_by
#         self.regression = regression
#         # self.convs = torch.nn.ModuleList()
#         # self.convs.append(RGCNConv(dataset.num_features, latent_dim[0], num_relations, num_bases))
#         # for i in range(0, len(latent_dim)-1):
#         #     self.convs.append(RGCNConv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        

#         self.encoder = MPNEncoder(atom_fdim = dataset.num_features, 
#                        bond_fdim = 1,
#                        hidden_channels = hidden_channels,
#                        bias = bias,
#                        depth = depth,
#                        dropout = dropout,
#                        undirected = undirected)
#         self.lin1 = Linear(2*hidden_channels, 128)
#         self.side_features = side_features
#         if side_features:
#             self.lin1 = Linear(2*hidden_channels+n_side_features, 128)

#         if self.regression:
#             self.lin2 = Linear(128, 1)
#         else:
#             self.lin2 = Linear(128, dataset.num_classes)

#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()

#     def forward(self, data):
#         x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        
#         if self.adj_dropout > 0:
#             edge_index, edge_type = dropout_adj(
#                 edge_index, edge_type, p=self.adj_dropout, 
#                 force_undirected=self.force_undirected, num_nodes=len(x), 
#                 training=self.training
#             )
#         out = self.encoder(x, edge_index, edge_type, batch)
        
#         users = data.x[:, 0] == 1
#         items = data.x[:, 1] == 1
#         # x = torch.cat([concat_states[users], concat_states[items]], 1)
#         x = torch.cat([out[users], out[items]], 1)
#         if self.side_features:
#             x = torch.cat([x, data.u_feature, data.v_feature], 1)

#         # x = torch.cat([x1, x2], dim=1)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         if self.regression:
#             return x[:, 0] * self.multiply_by
#         else:
#             return F.log_softmax(x, dim=-1)