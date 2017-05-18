'''
The structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 16th March 2017
'''

import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import ipdb


class HumanNodeRNN(nn.Module):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, args, infer=False):
        super(HumanNodeRNN, self).__init__()

        self.args = args
        self.infer = infer

        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.decoder_size = args.human_node_decoder_size
        self.input_size = args.human_node_input_size

        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.encoder_relu = nn.ReLU()

        if args.temporal:
            # Only temporal edges
            self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)
        elif args.noedges:
            # No edges at all
            self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)
        else:
            # Both spatial and temporal edges
            self.cell = nn.LSTMCell(3*self.embedding_size, self.rnn_size)

        self.decoder_linear = nn.Linear(self.rnn_size, self.decoder_size)
        self.decoder_relu = nn.ReLU()

        self.output_linear = nn.Linear(self.decoder_size, self.output_size)
        # self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, h_temporal, h_spatial_other, h, c):
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.encoder_relu(encoded_input)

        if self.args.noedges:
            # Only the encoded input
            concat_encoded = encoded_input
        elif self.args.temporal:
            # Concat only the temporal embedding
            concat_encoded = torch.cat((encoded_input, h_temporal), 1)
        else:
            # Concat both the embeddings
            concat_encoded = torch.cat((encoded_input, h_temporal, h_spatial_other), 1)

        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Decode hidden state
        out = self.decoder_linear(h_new)
        out = self.decoder_relu(out)

        # Get output
        # out = self.output_linear(h_new)
        out = self.output_linear(out)

        return out, h_new, c_new


class HumanHumanEdgeRNN(nn.Module):
    '''
    Class representing the Human-Human Edge RNN in the s-t graph
    '''
    def __init__(self, args, infer=False):
        super(HumanHumanEdgeRNN, self).__init__()

        self.args = args
        self.infer = infer

        self.rnn_size = args.human_human_edge_rnn_size
        self.embedding_size = args.human_human_edge_embedding_size
        self.input_size = args.human_human_edge_input_size

        self.encoder_linear_1 = nn.Linear(self.input_size, self.embedding_size)
        self.encoder_relu_1 = nn.ReLU()

        self.encoder_linear_2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.encoder_relu_2 = nn.ReLU()

        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):

        # Encode the input position
        encoded_input = self.encoder_linear_1(inp)
        encoded_input = self.encoder_relu_1(encoded_input)

        encoded_input = self.encoder_linear_2(encoded_input)
        encoded_input = self.encoder_relu_2(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new


class EdgeAttention(nn.Module):
    def __init__(self, args, infer=False):
        super(EdgeAttention, self).__init__()

        self.args = args
        self.infer = infer

        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size

        self.node_layer = nn.Linear(self.human_node_rnn_size, self.human_human_edge_rnn_size, bias=False)
        self.edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.human_human_edge_rnn_size, bias=False)

        # self.variable_length_layer = nn.Linear(self.)

        self.nonlinearity = nn.Tanh()
        self.general_weight_matrix = nn.Parameter(torch.Tensor(self.human_human_edge_rnn_size, self.human_human_edge_rnn_size))

        self.general_weight_matrix.data.normal_(0.1, 0.01)

    def forward(self, h_node, h_edges):
        '''
        h_node : Hidden state of the node at (tstep -1). Of size 1 x human_node_rnn_size
        h_edges : Hidden states of all spatial edges connected to the node at tstep. Of size n x human_human_edge_rnn_size
        '''
        num_edges = h_edges.size()[0]
        # Compute attention
        # Apply layers on top of edges and node
        node_embed = self.node_layer(h_node).squeeze(0)
        # edges_embed = self.edge_layer(h_edges)
        edges_embed = h_edges

        if self.args.attention_type == 'concat':
            # Concat based attention
            attn = node_embed.expand(num_edges, self.human_human_edge_rnn_size) + edges_embed
            attn = self.nonlinearity(attn)
            attn = torch.sum(attn, dim=1).squeeze(1)
        elif self.args.attention_type == 'dot':
            # Dot based attention
            attn = torch.mv(edges_embed, node_embed)
            # Variable length # NOTE multiplying the unnormalized weights with number of edges for now
            attn = torch.mul(attn, num_edges)
        else:
            # General attention
            attn = torch.mm(edges_embed, self.general_weight_matrix)
            attn = torch.mv(attn, node_embed)

        attn = torch.nn.functional.softmax(attn)
        # Compute weighted value
        weighted_value = torch.mv(torch.t(h_edges), attn)
        return weighted_value, attn


class SRNN(nn.Module):
    '''
    Class representing the SRNN model
    '''
    def __init__(self, args, infer=False):
        super(SRNN, self).__init__()

        self.args = args
        self.infer = infer

        if self.infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length

        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = HumanNodeRNN(args, infer)
        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal = HumanHumanEdgeRNN(args, infer)

        # Initialize attention module
        self.attn = EdgeAttention(args, infer)

    def forward(self, nodes, edges, nodesPresent, edgesPresent, hidden_states_node_RNNs, hidden_states_edge_RNNs,
                cell_states_node_RNNs, cell_states_edge_RNNs):
        '''
        Parameters
        ==========

        nodes : A tensor of shape seq_length x numNodes x 1 x 2
        Each row contains (x, y)

        edges : A tensor of shape seq_length x numNodes x numNodes x 1 x 2
        Each row contains the vector representing the edge
        If edge doesn't exist, then the row contains zeros

        nodesPresent : A list of lists, of size seq_length
        Each list contains the nodeIDs that are present in the frame

        edgesPresent : A list of lists, of size seq_length
        Each list contains tuples of nodeIDs that have edges in the frame

        hidden_states_node_RNNs : A tensor of size numNodes x 1 x node_rnn_size
        Contains hidden states of the node RNNs

        hidden_states_edge_RNNs : A tensor of size numNodes x numNodes x 1 x edge_rnn_size
        Contains hidden states of the edge RNNs

        Returns
        =======

        outputs : A tensor of shape seq_length x numNodes x 1 x 5
        Contains the predictions for next time-step

        hidden_states_node_RNNs

        hidden_states_edge_RNNs
        '''
        # Get number of nodes
        numNodes = nodes.size()[1]

        # Initialize output array
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size)).cuda()

        # Data structure to store attention weights
        attn_weights = [{} for _ in range(self.seq_length)]

        for framenum in range(self.seq_length):
            edgeIDs = edgesPresent[framenum]
            temporal_edges = [x for x in edgeIDs if x[0] == x[1]]
            spatial_edges = [x for x in edgeIDs if x[0] != x[1]]
            edges_current = edges[framenum]

            hidden_states_nodes_from_edges_temporal = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())
            hidden_states_nodes_from_edges_spatial = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())

            # Edges
            if len(edgeIDs) != 0 and (not self.args.noedges):

                # Temporal Edges
                if len(temporal_edges) != 0:

                    list_of_temporal_edges = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in edgeIDs if x[0] == x[1]]).cuda())
                    list_of_temporal_nodes = torch.LongTensor([x[0] for x in edgeIDs if x[0] == x[1]]).cuda()

                    edges_temporal_start_end = torch.index_select(edges_current, 0, list_of_temporal_edges)
                    hidden_temporal_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges)
                    cell_temporal_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges)

                    h_temporal, c_temporal = self.humanhumanEdgeRNN_temporal(edges_temporal_start_end, hidden_temporal_start_end,
                                                                             cell_temporal_start_end)
                    # ipdb.set_trace()

                    hidden_states_edge_RNNs[list_of_temporal_edges.data] = h_temporal
                    cell_states_edge_RNNs[list_of_temporal_edges.data] = c_temporal

                    hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes] = h_temporal

                # Spatial Edges
                if len(spatial_edges) != 0 and (not self.args.temporal):

                    list_of_spatial_edges = Variable(torch.LongTensor([x[0]*numNodes + x[1] for x in edgeIDs if x[0] != x[1]]).cuda())
                    list_of_spatial_nodes = np.array([x[0] for x in edgeIDs if x[0] != x[1]])

                    edges_spatial_start_end = torch.index_select(edges_current, 0, list_of_spatial_edges)
                    hidden_spatial_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_spatial_edges)
                    cell_spatial_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_spatial_edges)

                    h_spatial, c_spatial = self.humanhumanEdgeRNN_spatial(edges_spatial_start_end, hidden_spatial_start_end,
                                                                          cell_spatial_start_end)

                    hidden_states_edge_RNNs[list_of_spatial_edges.data] = h_spatial
                    cell_states_edge_RNNs[list_of_spatial_edges.data] = c_spatial

                    # Sum Spatial
                    if self.args.temporal_spatial:
                        for node in range(numNodes):
                            l = torch.LongTensor(np.where(list_of_spatial_nodes == node)[0]).cuda()
                            if torch.numel(l) == 0:
                                continue
                            hidden_states_nodes_from_edges_spatial[node] = torch.sum(h_spatial[l], 0)

                    # Attention
                    else:
                        for node in range(numNodes):
                            l = torch.LongTensor(np.where(list_of_spatial_nodes == node)[0]).cuda()
                            node_others = [x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]]
                            if torch.numel(l) == 0:
                                continue
                            ind = Variable(torch.LongTensor([node]).cuda())
                            h_node = torch.index_select(hidden_states_node_RNNs, 0, ind)
                            hidden_attn_weighted, attn_w = self.attn(h_node, h_spatial[l])
                            attn_weights[framenum][node] = (attn_w.data.cpu().numpy(), node_others)
                            hidden_states_nodes_from_edges_spatial[node] = hidden_attn_weighted

            # Nodes
            nodeIDs = nodesPresent[framenum]

            if len(nodeIDs) != 0:

                list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())

                nodes_current = torch.index_select(nodes[framenum], 0, list_of_nodes)

                hidden_nodes_current = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes)
                cell_nodes_current = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes)

                h_temporal_other = hidden_states_nodes_from_edges_temporal[list_of_nodes.data]
                h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]

                outputs[framenum * numNodes + list_of_nodes.data], h_nodes, c_nodes = self.humanNodeRNN(nodes_current, h_temporal_other, h_spatial_other, hidden_nodes_current, cell_nodes_current)

                hidden_states_node_RNNs[list_of_nodes.data] = h_nodes
                cell_states_node_RNNs[list_of_nodes.data] = c_nodes

        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size).cuda())
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs, attn_weights
