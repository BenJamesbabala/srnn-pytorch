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
        self.input_embedding_size = args.human_node_embedding_size
        self.tensor_embedding_size = args.human_tensor_embedding_size
        self.input_size = args.human_node_input_size

        self.encoder_linear = nn.Linear(self.input_size, self.input_embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.tensor_embed = nn.Linear(self.rnn_size, self.tensor_embedding_size)

        self.cell = nn.LSTMCell(self.input_embedding_size + self.tensor_embedding_size, self.rnn_size)

        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, h_spatial_other, h, c):
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # Concat both the embeddings
        h_tensor = h_spatial_other
        h_tensor_embedded = self.relu(self.tensor_embed(h_tensor))
        h_tensor_embedded = self.dropout(h_tensor_embedded)
        concat_encoded = torch.cat((encoded_input, h_tensor_embedded), 1)
        
        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        out = self.output_linear(h_new)

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

        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):

        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

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

        self.node_layer = nn.Linear(self.human_node_rnn_size, self.human_human_edge_rnn_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        
        self.nonlinearity = nn.Tanh()
        self.general_weight_matrix = nn.Parameter(torch.Tensor(self.human_human_edge_rnn_size, self.human_human_edge_rnn_size))

        self.general_weight_matrix.data.normal_(0.1, 0.01)

    def forward(self, h_node, h_nodes_other):
        '''
        h_nodes : Hidden states of all the nodes at (tstep -1). Of size m x human_node_rnn_size, where m
        is the number of nodes at tstep
        '''
        num_nodes_other = h_nodes_other.size()[0]

        # First, embed the current hidden state
        node_embed = h_node.squeeze(0)
        # Then, embed all the other hidden states
        node_embed_other = h_nodes_other

        attn = torch.mv(node_embed_other, node_embed)
        temperature = num_nodes_other
        attn = torch.mul(attn, temperature)

        attn = torch.nn.functional.softmax(attn)
        # Compute weighted value
        weighted_value = torch.mv(torch.t(node_embed_other), attn)
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

            # hidden_states_nodes_from_edges_temporal = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())
            # hidden_states_nodes_from_edges_spatial = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())

            # Nodes
            nodeIDs = nodesPresent[framenum]
            numNodesCurrent = len(nodeIDs)

            if len(nodeIDs) != 0:

                list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())

                nodes_current = torch.index_select(nodes[framenum], 0, list_of_nodes)

                hidden_nodes_current = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes)
                cell_nodes_current = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes)

                # h_temporal_other = hidden_states_nodes_from_edges_temporal[list_of_nodes.data]                
                # h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]
                h_spatial_other = Variable(torch.zeros(numNodesCurrent, self.human_node_rnn_size).cuda())
                
                if numNodesCurrent > 1:
                    for node in range(numNodesCurrent):
                        node_other = [x for x in range(numNodesCurrent) if x != node]
                        list_of_other_nodes = Variable(torch.LongTensor(node_other).cuda())
                        h_current = hidden_nodes_current[node]
                        h_other = torch.index_select(hidden_nodes_current, 0, list_of_other_nodes)
                        h_spatial, attn_w = self.attn(h_current.view(1, self.human_node_rnn_size), h_other)
                        h_spatial_other[node] = h_spatial                        
                        attn_weights[framenum][nodeIDs[node]] = (attn_w.data.cpu().numpy(), [nodeIDs[x] for x in node_other])
                        

                outputs[framenum * numNodes + list_of_nodes.data], h_nodes, c_nodes = self.humanNodeRNN(nodes_current, h_spatial_other, hidden_nodes_current, cell_nodes_current)

                hidden_states_node_RNNs[list_of_nodes.data] = h_nodes
                cell_states_node_RNNs[list_of_nodes.data] = c_nodes

        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size).cuda())
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs, attn_weights
