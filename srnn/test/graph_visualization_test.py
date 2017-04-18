from ..visualize import make_dot
from ..model import SRNN
from ..utils import DataLoader
from ..st_graph import ST_GRAPH
import pickle
import os
import torch
from torch.autograd import Variable

with open(os.path.join('save', 'config.pkl'), 'rb') as f:
    args = pickle.load(f)

net = SRNN(args)

dataset = [0]

dataloader = DataLoader(1, 8, dataset, True)

dataloader.reset_batch_pointer()

stgraph = ST_GRAPH(1, 8)

x, _, d = dataloader.next_batch()
stgraph.readGraph(x)

nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence(0)

# Convert to cuda variables
nodes = Variable(torch.from_numpy(nodes).float()).cuda()
edges = Variable(torch.from_numpy(edges).float()).cuda()

numNodes = nodes.size()[1]
hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()
cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

net.zero_grad()

outputs, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)

print outputs
