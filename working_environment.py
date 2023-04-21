"""
@author: Stephen Lasinis
"""
#Import our Graph Utility functions
import GraphUtil as gu
import GraphDraw as gd
from math import trunc
    
A = [[0,1,0,0,1,1,0,0,0,0],
     [1,0,1,0,0,0,1,0,0,0],
     [0,1,0,1,0,0,0,1,0,0],
     [0,0,1,0,1,0,0,0,1,0],
     [1,0,0,1,0,0,0,0,0,1],
     [1,0,0,0,0,0,0,1,1,0],
     [0,1,0,0,0,0,0,0,1,1],
     [0,0,1,0,0,1,0,0,0,1],
     [0,0,0,1,0,1,1,0,0,0],
     [0,0,0,0,1,0,1,1,0,0]]

G = gu.create_graph_custom(A)
#print(gu.pinnaclus_brutus(G))

gu.time_analysis(gu.create_graph(11,'bipartite6'))

# print(gu.pinnaclus_brutus(gu.create_graph(8,'cycle')))
# print(gu.pinnacle_computation(gu.GraphType.CYCLE,[8],8))