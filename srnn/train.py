'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import time
import numpy as np

import torch
from torch.autograd import Variable

from utils import DataLoader
from st_graph import ST_GRAPH
from model import SRNN
from criterion import Gaussian2DLikelihood


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
    parser.add_argument('--human_human_edge_input_size', type=int, default=3,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=5,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=8,
                        help='Sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=1.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')

    args = parser.parse_args()
    train(args)


def train(args):
    datasets = range(4)
    # Remove the leave out dataset from the datasets
    datasets.remove(args.leaveDataset)

    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length + 1, datasets, forcePreProcess=True)

    # Construct the ST-graph object
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + 1)

    with open(os.path.join('save', 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    def checkpoint_path(x):
        return os.path.join('save', 'srnn_model_'+str(x)+'.tar')

    net = SRNN(args)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    learning_rate = args.learning_rate
    print 'Training begin'
    # Training
    for epoch in range(args.num_epochs):
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # learning_rate *= args.decay_rate
        learning_rate /= np.sqrt(epoch + 1)

        dataloader.reset_batch_pointer()

        for batch in range(dataloader.num_batches):
            start = time.time()

            # TODO Modify dataloader so that it doesn't return separate source and target data
            # TODO Also, make sure each batch comes from the same dataset
            x, _, d = dataloader.next_batch()

            # Read the st graph from data
            stgraph.readGraph(x)

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence(sequence)

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                # Zero out the gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                          hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                          cell_states_node_RNNs, cell_states_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:])

                loss_batch += loss.data[0]

                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            stgraph.reset()
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

            if ((epoch * dataloader.num_batches + batch) % args.save_every == 0 and ((epoch * dataloader.num_batches + batch) > 0)) or (epoch * dataloader.num_batches + batch + 1 == args.num_epochs * dataloader.num_batches):
                print 'Saving model'
                torch.save({
                    'epoch': epoch,
                    'batch': batch,
                    'iteration': epoch*dataloader.num_batches + batch,
                    'state_dict': net.state_dict()
                }, checkpoint_path(epoch*dataloader.num_batches + batch))

if __name__ == '__main__':
    main()
