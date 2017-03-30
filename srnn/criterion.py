'''
Criterion for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 30th March 2017
'''


import torch
import numpy as np


def Gaussian2DLikelihood(outputs, targets, nodesPresent):
    '''
    Parameters:

    outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0

    for framenum in range(outputs.size()[0]):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:

            loss = loss + result[framenum, nodeID]

    return loss
