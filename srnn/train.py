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
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
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
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=256,
                        help='Embedding size of edge features')

    # Decoder size
    parser.add_argument('--human_node_decoder_size', type=int, default=50,
                        help='Number of hidden units in the decoder layer')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=20,
                        help='Sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    
    parser.add_argument('--patience', type=int, default=30,
                        help='Patience')

    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=200,
                        help='save frequency')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.96,
                        help='decay rate for rmsprop')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')

    # Experiments
    parser.add_argument('--noedges', action='store_true')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--temporal_spatial', action='store_true')
    parser.add_argument('--attention', action='store_true')

    # Attention type
    parser.add_argument('--attention_type', choices=['concat', 'dot', 'general'], type=str, default='dot',
                        help='Attention type')

    args = parser.parse_args()

    # Check experiment tags
    if not (args.noedges or args.temporal or args.temporal_spatial or args.attention):
        print 'Use one of the experiment tags to enforce model'
        return

    train(args)


def train(args):
    # datasets = range(4)
    # Remove the leave out dataset from the datasets
    # datasets.remove(args.leaveDataset)
    datasets = [0, 1, 3]
    args.leaveDataset = 2

    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length + 1, datasets, forcePreProcess=True)

    # Construct the ST-graph object
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + 1)

    # Log directory
    log_directory = 'log/'
    log_directory += str(args.leaveDataset)+'/'
    if args.noedges:
        print 'No edge RNNs used'
        log_directory += 'log_noedges'
    elif args.temporal:
        print 'Only temporal edge RNNs used'
        log_directory += 'log_temporal'
    elif args.temporal_spatial:
        print 'Both temporal and spatial edge RNNs used'
        log_directory += 'log_temporal_spatial'
    else:
        print 'Both temporal and spatial edge RNNs used with attention'
        log_directory += 'log_attention'

    # Logging file
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')
    
    # Save directory
    save_directory = 'save/'
    save_directory += str(args.leaveDataset)+'/'
    if args.noedges:
        print 'No edge RNNs used'
        save_directory += 'save_noedges'
    elif args.temporal:
        print 'Only temporal edge RNNs used'
        save_directory += 'save_temporal'
    elif args.temporal_spatial:
        print 'Both temporal and spatial edge RNNs used'
        save_directory += 'save_temporal_spatial'
    else:
        print 'Both temporal and spatial edge RNNs used with attention'
        save_directory += 'save_attention'

    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    def checkpoint_path(x):
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    net = SRNN(args)
    net.cuda()

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)
    learning_rate = args.learning_rate
    print 'Training begin'
    best_val_loss = 100
    best_epoch = 0
    # Training
    for epoch in range(args.num_epochs):
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        learning_rate *= args.decay_rate
        # learning_rate = args.learning_rate / np.sqrt(epoch + 1)

        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # Training
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, _, d = dataloader.next_batch(randomUpdate=True)

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
                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
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
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

            '''
            if ((epoch * dataloader.num_batches + batch) % args.save_every == 0 and ((epoch * dataloader.num_batches + batch) > 0)) or (epoch * dataloader.num_batches + batch + 1 == args.num_epochs * dataloader.num_batches):
                print 'Saving model'
                torch.save({
                    'epoch': epoch,
                    'batch': batch,
                    'iteration': epoch*dataloader.num_batches + batch,
                    'state_dict': net.state_dict()
                }, checkpoint_path(epoch*dataloader.num_batches + batch))
            '''
        loss_epoch /= dataloader.num_batches
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')
        print '*************'
        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        for batch in range(dataloader.valid_num_batches):
            # Get batch data
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

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

                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                             hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                             cell_states_node_RNNs, cell_states_edge_RNNs)                

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:])

                loss_batch += loss.data[0]

            stgraph.reset()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch = loss_epoch / dataloader.valid_num_batches

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch

        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        print 'Best epoch', best_epoch, 'Best validation loss', best_val_loss
        log_file_curve.write(str(loss_epoch)+'\n')
        print '*************'

        # Save the model after each epoch
        print 'Saving model'
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    print 'Best epoch', best_epoch, 'Best validation Loss', best_val_loss
    log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()

if __name__ == '__main__':
    main()
