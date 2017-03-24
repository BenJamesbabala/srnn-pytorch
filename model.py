'''
The structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 16th March 2017
'''

import torch.nn as nn
from torch.autograd import Variable
import torch


class HumanRNN(nn.module):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, args, infer=False):
        super(HumanRNN, self).__init__()

        self.args = args
        self.infer = infer

        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size

        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.encoder_relu = nn.ReLU()

        self.hidden_encoder_linear = nn.Linear(self.input_size, self.embedding_size)
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
        concat_encoded = torch.cat((encoded_input, encoded_hidden), dimension=1)

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

        self.args = args
        self.infer = infer

        self.rnn_size = args.human_human_edge_rnn_size
        self.embedding_size = args.human_human_edge_embedding_size
        self.input_size = args.human_human_edge_input_size

        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.encoder_relu = nn.ReLU()

        self.hidden_encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.hidden_encoder_relu = nn.ReLU()

        self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)

    def init_weights(self):

        self.encoder_linear.weight.data.normal_(0, 0.1)
        self.encoder_linear.bias.data.fill_(0)

        self.hidden_encoder_linear.weight.data.normal_(0, 0.1)
        self.hidden_encoder_linear.bias.data.fill_(0)

    def forward(self, inp, h_other, h):

        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.encoder_relu(encoded_input)

        # Encode the input hidden states
        encoded_hidden = self.hidden_encoder_linear(h_other)
        encoded_hidden = self.hidden_encoder_relu(encoded_hidden)

        # Concat both the embeddings
        concat_encoded = torch.cat((encoded_input, encoded_hidden), dimension=1)

        # One-step of GRU
        h_new = self.cell(concat_encoded, h)

        return h_new
