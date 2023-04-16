"""
@author: Stephen Lasinis
"""
#Import our Graph Utility functions
import GraphUtil as gu
import GraphDraw as gd

g_list = gu.pinnaclus_utopius(gu.create_graph(5,'star1'),[5,4],True,True)[1]

gd.draw_graphs(g_list)