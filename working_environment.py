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

# u = gu.pinnaclus_utopius(gu.create_graph(11,'wheel'),[11],True)
# print(u[0])
# print(len(u[1]))
# print(gu.pinnaclus_brutus(gu.create_graph(11,'wheel'),True)['[11]'])
print(gu.pinnacle_computation(gu.GraphType.STAR,[15,14,13,12,11],15,star_count=5,time_log=True))