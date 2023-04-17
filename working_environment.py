"""
@author: Stephen Lasinis
"""
#Import our Graph Utility functions
import GraphUtil as gu
import GraphDraw as gd

total,g_list = gu.pinnaclus_utopius(gu.create_graph(9,'wheel'),[9,8,7],True,complete=True)
print(total)
gd.draw_graphs(g_list,100)