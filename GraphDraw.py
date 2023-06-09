from graphviz import *
import os
import GraphUtil as gu
from math import trunc
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

def draw_graphs(graph_list: list, max_len: int = 25):
    
    if len(graph_list) > max_len:
        val = trunc(len(graph_list) / max_len)
        graph_list = [g for i,g in enumerate(graph_list) if i % val == 0 ]
    
    G = Digraph(engine='dot',name='cluster_graph')
    for j,g in enumerate(graph_list):
        P = g.get_pinnacles()
        sG = Digraph(name=f'cluster_{g}')
        for i,n in enumerate(g.node_list):
            if n.value in P: col = 'red'
            else: col = 'blue'
            if n.value == 0: n.set_value(1)
            sG.node(name=f'{j},{i}',label=f'{n.value}', color=col)
        for r in range(len(g.adjacency_matrix)):
            for c in range(r+1,len(g.adjacency_matrix[r])):
                if g.adjacency_matrix[r][c] != 0: sG.edge(f'{j},{r}',f'{j},{c}', dir='none')
        G.subgraph(sG)
    
    G.render(view=True)
