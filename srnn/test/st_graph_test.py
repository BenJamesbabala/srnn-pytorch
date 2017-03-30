from ..utils import DataLoader
from ..st_graph import ST_GRAPH

bsize = 100
slength = 10

dataloader = DataLoader(batch_size=bsize, seq_length=slength, datasets=[0], forcePreProcess=True)
x, y, d = dataloader.next_batch()

graph = ST_GRAPH(batch_size=bsize, seq_length=slength)
graph.readGraph(x)

# graph.printGraph()

nodes, edges, nodesPresent, edgesPresent = graph.getSequence(0)
