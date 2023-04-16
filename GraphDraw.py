from graphviz import *
import os
import GraphUtil as gu
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

def draw_graphs(graph_list: list):
    graph_list = [gu.create_graph(5,'star2'), gu.create_graph(7,'bipartite4')]
    G = Graph()
    for g in graph_list:
        sG = Graph()
        for n in g.node_list:
            sG.node('n', f'{n.get_value()}')

G = Graph()
G.node('1', '1')
G.node('2', '2')
G.node('3', '3')
G.node('4','4',color='red')

G.edges(['12','13','23','14','24','34'])

G.render(view=True)