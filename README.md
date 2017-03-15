# Structural RNN using PyTorch

This code/implementation is available for research purposes. If you are using this code for your work, please give due credits.


## Implementation Plan

### S-T graph
- Must contain stuff relevant to the s-t graph for a fixed-length time sequence
- Should take in as input, consecutive video frames with annotations and return a s-t graph where the nodes are humans (and in the future, static obstacles)
- Rather than calling it for each sequence, design it such that it can take a minibatch of data (or the entire data)

### Node RNN
- A parent class that encapsulates all aspects of a node in SRNN

### Human RNN
- Must extend the node RNN class and include an encoder-decoder seq2seq model 
- The input of RNN must be the current position of the human and the sum of hidden states of all the edge RNNs connected to this node
- The output must be the position at next time-step

### Obstacle RNN
- Must extend the node RNN class and include an encoder RNN model
- The input of RNN should be the sum of hidden states of all edge RNNs connected to this node
- No output (only the hidden state is used)

### Edge RNN
- A parent class that encapsulates all aspects of a edge in SRNN

### Human-Human RNN
- Must extend the edge RNN class and include an encoder RNN model
- The input of RNN should be a relative orientation vector of the humans involved (something in absolute scale) with the distance. (Or something else, should be designed carefully)
- Additional inputs involve hidden states of the human node RNNs involved with the edge
- No output (only the hidden state is used)

### Human-Obstacle RNN
- Must extend the edge RNN class and include an encoder RNN model
- The input of RNN should be the same as Human-Human RNN
- Additional inputs involve hidden states of the human node RNN and the obstacle node RNN involved with the edge
- No output (only the hidden state is used)

**Author** : Anirudh Vemula

**Affiliation** : Robotics Institute, Carnegie Mellon University

**License** : GPL v3
