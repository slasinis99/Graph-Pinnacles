from __future__ import annotations
from enum import Enum, auto
import time

#########
# ENUMS #
#########

class GraphType(Enum):
    STAR = auto()
    COMPLETE = auto()
    CYCLE = auto()
    WHEEL = auto()
    BIPARTITE = auto()
    CUSTOM = auto()

###########
# CLASSES #
###########

class Node():
    

    def __init__(self, value: int = 0):
        self.value = value
        self.connections = set()
    
    def __str__(self):
        return f'Value = {self.value}'
    
    def set_value(self, value: int) -> set:
        self.value = value
        return self.connections
    
    def add_connection(self, node: Node):
        self.connections.add(node)

class Graph():


    def __init__(self, adjacency_matrix: list[list], type: GraphType = GraphType.CUSTOM, id: str = ''):
        self.id = id
        self.type = type
        self.matrix = adjacency_matrix
        self.reach = [set(), set()]
        self.nodes = [Node() for _ in range(len(adjacency_matrix))]
        for r, node in enumerate(self.nodes):
            for c, other_node in enumerate(self.nodes):
                if self.matrix[r][c]: node.add_connection(other_node)
    
    def __str__(self):
        s = ''
        if self.id != '': s += f'ID: {self.id}\n'
        s += f'Graph Type: {self.type}\n'
        for i, node in enumerate(self.nodes):
            s += f'Node {i+1}: {str(node)}\n'
        return s
    
    def set_node_value(self, node: Node, value: int):
        self.reach[0].add(node)
        self.reach[1] = self.reach[1] | node.set_value(value)
        if node in self.reach[1]: self.reach[1].remove(node)

    def reset_node_values(self):
        for node in self.nodes:
            node.value = 0
        self.reach = [set(), set()]

################
# PERMUTATIONS #
################

def generate_permutations(size: int) -> list:
    """Return a list of all permutations of the given size using values [1,...,size]"""
    
    S = []
    _generate_permutations(S, [i+1 for i in range(size)], size)
    return S

def _generate_permutations(S: list, a: list, size: int):
    """Generate the permutations of a onto S recursively"""
    
    if (size == 1):
        S.append(a.copy())
        return
    
    for i in range(size):
        _generate_permutations(S, a, size-1)
        
        if (size & 1):
            a[0], a[size-1] = a[size-1], a[0]
        else: a[i], a[size-1] = a[size-1], a[i]    

############################
# Fast Pinnacle Generation #
############################

def get_nontouching_nodes(G: Graph, n: int, time_log: bool = False) -> list:


    if time_log: t = time.time()

    pairings = [[[{node},node.connections] for node in G.nodes]]

    while len(pairings) < n:
        new_pairs = []
        pair_dict = {}
        for n1 in pairings[-1]:
            for n2 in pairings[0]:
                if not any(val in n1[0] for val in n2[0]) and not any(val in n1[1] for val in n2[0]) and not n1[0] | n2[0] in pair_dict.values():
                    new_pairs.append([n1[0] | n2[0], n1[1] | n2[1]])
                    pair_dict[str(len(new_pairs))] = n1[0] | n2[0]
        pairings.append(new_pairs)
    
    if time_log: print(f'get_nontouching_pairs runtime = {time.time()-t}secs')

    return pairings[-1]


###########
# TESTING #
###########

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

G = Graph(A, GraphType.CUSTOM, 'My Graph')

print(len(get_nontouching_nodes(G,2,True)))