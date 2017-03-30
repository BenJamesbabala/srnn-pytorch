from ..criterion import Gaussian2DLikelihood
import torch
from torch.autograd import Variable


outputs = torch.ones(5, 5, 5)
targets = torch.zeros(5, 5, 2)
nodesPresent = [[0, 1], [1, 2], [1, 2, 3], [3, 4], [3, 4]]

outputs = Variable(outputs)
targets = Variable(targets)

loss = Gaussian2DLikelihood(outputs, targets, nodesPresent)
print loss
