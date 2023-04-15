"""
@author: Stephen Lasinis
"""
#Import our Graph Utility functions
import GraphUtil as gu

G = gu.createGraph(20,style='star5')

pair = gu.pinnaclusPairalingus(G,5)
print(len(pair))

#gu.verifyBipartite(6,5)
