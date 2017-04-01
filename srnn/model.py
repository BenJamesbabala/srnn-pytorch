'''
The structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 16th March 2017
'''

import torch.nn as nn
from torch.autograd import Variable
import torch


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
        self.input_size = args.human_node_input_size

        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.encoder_relu = nn.ReLU()

        self.hidden_encoder_linear = nn.Linear(args.human_human_edge_rnn_size*2, self.embedding_size)
        self.hidden_encoder_relu = nn.ReLU()

        self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)

        # self.lr = args.learning_rate

        self.decoder_linear = nn.Linear(self.rnn_size, self.output_size)

    def init_weights(self):
        self.encoder_linear.weight.data.normal_(0, 0.1)
        self.encoder_linear.bias.data.fill_(0)

        self.hidden_encoder_linear.weight.data.normal_(0, 0.1)
        self.hidden_encoder_linear.bias.data.fill_(0)

        self.decoder_linear.weight.data.normal_(0, 0.1)
        self.decoder_linear.bias.data.fill_(0)

    def forward(self, pos, h_other, h):
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.encoder_relu(encoded_input)

        # Encode the input hidden states
        encoded_hidden = self.hidden_encoder_linear(h_other)
        encoded_hidden = self.hidden_encoder_relu(encoded_hidden)

        # Concat both the embeddings
        concat_encoded = torch.cat((encoded_input, encoded_hidden), 1)

        # One-step of GRU
        h_new = self.cell(concat_encoded, h)

        # Decode hidden state
        out = self.decoder_linear(h_new)

        return out, h_new


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
        self.encoder_relu = nn.ReLU()

        # self.hidden_encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        # self.hidden_encoder_relu = nn.ReLU()

        self.cell = nn.GRUCell(self.embedding_size, self.rnn_size)

    def init_weights(self):

        self.encoder_linear.weight.data.normal_(0, 0.1)
        self.encoder_linear.bias.data.fill_(0)

        # self.hidden_encoder_linear.weight.data.normal_(0, 0.1)
        # self.hidden_encoder_linear.bias.data.fill_(0)

    def forward(self, inp, h):

        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.encoder_relu(encoded_input)

        # Encode the input hidden states
        # encoded_hidden = self.hidden_encoder_linear(h_other)
        # encoded_hidden = self.hidden_encoder_relu(encoded_hidden)

        # Concat both the embeddings
        # concat_encoded = torch.cat((encoded_input, encoded_hidden), dimension=0)

        # One-step of GRU
        h_new = self.cell(encoded_input, h)

        return h_new


class SRNN(nn.Module):
    '''
    Class representing the SRNN model
    '''
    def __init__(self, args, infer=False):
        super(SRNN, self).__init__()

        self.args = args
        self.infer = infer

        self.seq_length = args.seq_length
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = HumanNodeRNN(args, infer)
        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal = HumanHumanEdgeRNN(args, infer)

        # Initialize the weights of the Node and Edge RNNs
        self.humanNodeRNN.init_weights()
        self.humanhumanEdgeRNN_spatial.init_weights()
        self.humanhumanEdgeRNN_temporal.init_weights()

    def forward(self, nodes, edges, nodesPresent, edgesPresent):
        '''
        Parameters

        nodes : A tensor of shape seq_length x numNodes x 2
        Each row contains (x, y)

        edges : A tensor of shape seq_length x numNodes x numNodes x 2
        Each row contains the vector representing the edge
        If edge doesn't exist, then the row contains zeros

        nodesPresent : A list of lists, of size seq_length
        Each list contains the nodeIDs that are present in the frame

        edgesPresent : A list of lists, of size seq_length
        Each list contains tuples of nodeIDs that have edges in the frame
        '''
        # Get number of nodes
        numNodes = nodes.size()[1]

        # Initialize hidden states of node RNNs and edge RNNs
        hidden_states_node_RNNs = Variable(torch.zeros(numNodes, 1, self.human_node_rnn_size)).cuda()
        hidden_states_edge_RNNs = Variable(torch.zeros(numNodes, numNodes, 1, self.human_human_edge_rnn_size)).cuda()

        # Initialize output array
        outputs = Variable(torch.zeros(self.seq_length, numNodes, 1, self.output_size)).cuda()

        for framenum in range(self.seq_length):
            edgeIDs = edgesPresent[framenum]
            for edgeID in edgeIDs:
                # Distinguish between temporal and spatial edge
                if edgeID[0] == edgeID[1]:
                    # Temporal edge
                    nodeID = edgeID[0]
                    hidden_states_edge_RNNs[nodeID, nodeID, :] = self.humanhumanEdgeRNN_temporal(edges[framenum, nodeID, nodeID, :].view(1, -1), hidden_states_edge_RNNs[nodeID, nodeID, :].clone())
                else:
                    # Spatial edge
                    nodeID_a = edgeID[0]
                    nodeID_b = edgeID[1]
                    hidden_states_edge_RNNs[nodeID_a, nodeID_b, :] = self.humanhumanEdgeRNN_spatial(edges[framenum, nodeID_a, nodeID_b, :].view(1, -1), hidden_states_edge_RNNs[nodeID_a, nodeID_b, :].clone())

            nodeIDs = nodesPresent[framenum]

            for nodeID in nodeIDs:
                # Get edges corresponding to the node
                edgeIDs = [x for x in edgesPresent[framenum] if x[0] == nodeID]
                # Differentiate between temporal and spatial edge
                spatial_edgeIDs = [x for x in edgeIDs if x[0] != x[1]]
                # TODO : Simple addition for now
                h_spatial = Variable(torch.zeros(1, self.human_human_edge_rnn_size)).cuda()
                for edgeID in spatial_edgeIDs:
                    h_spatial = h_spatial + hidden_states_edge_RNNs[nodeID, edgeID[1], :]

                h_temporal = Variable(torch.zeros(1, self.human_human_edge_rnn_size)).cuda()
                h_temporal = h_temporal + hidden_states_edge_RNNs[nodeID, nodeID, :]

                h_other = torch.cat((h_temporal, h_spatial), 1)

                outputs[framenum, nodeID, :], hidden_states_node_RNNs[nodeID, :] = self.humanNodeRNN(nodes[framenum, nodeID, :].view(1, -1), h_other, hidden_states_node_RNNs[nodeID, :].clone())

        return outputs
