"""
@author: Stephen Lasinis
"""
#Import our Graph Utility functions
import GraphUtil as gu
#import GraphDraw as gd
from math import trunc
    
# A = [[0,1,0,0,1,1,0,0,0,0],
#      [1,0,1,0,0,0,1,0,0,0],
#      [0,1,0,1,0,0,0,1,0,0],
#      [0,0,1,0,1,0,0,0,1,0],
#      [1,0,0,1,0,0,0,0,0,1],
#      [1,0,0,0,0,0,0,1,1,0],
#      [0,1,0,0,0,0,0,0,1,1],
#      [0,0,1,0,0,1,0,0,0,1],
#      [0,0,0,1,0,1,1,0,0,0],
#      [0,0,0,0,1,0,1,1,0,0]]

# G = gu.create_graph_custom(A)
# print(f'Pinnacle Set Frequency for Petersen Graph')
# print(gu.pinnaclus_brutus(G))

#print(gu.pinnaclus_utopius(gu.create_graph(10,'wheel'),[10],True)[0])
print(gu.pinnaclus_brutus(gu.create_graph(9,'wheel'),True)['[9]'])
#print(gu.pinnacle_computation(gu.GraphType.BIPARTITE,[50,49,48,47,46],50,bipartite_left=30,time_log=True))