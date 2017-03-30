'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle

from utils import DataLoader
from st_graph import ST_GRAPH
from model import SRNN


def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=64,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=2,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=5,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=32,
                        help='Embedding size of edge features')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.005,
                        help='L2 regularization parameter')

    args = parser.parse_args()
    train(args)


def train(args):
    datasets = range(4)
    # Remove the leave out dataset from the datasets
    datasets.remove(args.leaveDataset)

    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length, datasets, forcePreProcess=True)

    with open(os.path.join('save', 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    srnn = SRNN(args)
    srnn.cuda()

    
