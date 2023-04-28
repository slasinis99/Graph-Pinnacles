from __future__ import annotations
from enum import Enum, auto
import time
from math import factorial

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
        self.size = len(adjacency_matrix)
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
        if not node in self.nodes: print('Node not in Graph')
        self.reach[0].add(node)
        self.reach[1] = self.reach[1] | node.set_value(value)
        for nodes in self.reach[0]:
            if nodes in self.reach[1]: self.reach[1].remove(nodes)

    def reset_node_values(self):
        for node in self.nodes:
            node.value = 0
        self.reach = [set(), set()]

    def copy(self) -> Graph:
        new_graph = Graph(self.matrix,self.type,self.id+'-Copy')
        for i, node in enumerate(self.nodes):
            if node.value != 0:
                new_graph.set_node_value(new_graph.nodes[i],node.value)
        return new_graph

##################
# GRAPH CREATION #     
##################

def digits_only(s: str) -> int:
    """Takes a string and returns an integer for the digits in string."""
    i = ''
    for d in [char for char in s if char.isdigit()]:
        i += d
    return int(i)

def create_graph(node_amount: int, style: str = 'star-1', id: str = '') -> Graph:
    """Create a Graph with the desired amount of nodes for given style
    
    Attributes:
        node_amount: int -- total number of nodes / vertices in the Graph
        style: str -- string that determines that specific of the graph type
    """
    
    if style[0:4] == 'star':
        return _star(node_amount=node_amount, star_amount=digits_only(style), id=id)
    elif style[0:9] == 'bipartite':
        return _bipartite(node_amount=node_amount, left_amount=digits_only(style), id=id)
    elif style[0:5] == 'wheel':
        return _wheel(node_amount=node_amount, id=id)
    elif style[0:5] == 'cycle':
        return _cycle(node_amount=node_amount, id=id)
    elif style[0:8] == 'complete':
        return _complete(node_amount=node_amount, id=id)

def create_graph_custom(adjacency_matrix: list, id: str = ''):
    return Graph(adjacency_matrix,GraphType.CUSTOM, id)

def _star(node_amount: int, star_amount: int, id: str = '') -> Graph:
    """Create a Star Graph Instance"""
    
    adjacency_matrix = []
    for i in range(node_amount):
        if i < star_amount:
            node = [1]*node_amount
            node[i] = 0
        else: node = [1]*star_amount + [0]*(node_amount-star_amount)
        adjacency_matrix.append(node)
    return Graph(adjacency_matrix, GraphType.STAR, id) 

def _bipartite(node_amount: int, left_amount: int, id: str = ''):
    """Create a Complete Bipartite"""

    adjacency_matrix = []
    for i in range(left_amount):
        adjacency_matrix.append([0]*left_amount + [1]*(node_amount-left_amount))
        
    for i in range(node_amount-left_amount):
        adjacency_matrix.append([1]*left_amount + [0]*(node_amount-left_amount))
        
    return Graph(adjacency_matrix, GraphType.BIPARTITE, id)

def _wheel(node_amount: int, id: str = '') -> Graph:
    """Returns a wheel graph with desired node amount."""
    
    adjacency_matrix = []
    
    for i in range(node_amount-1):
        node = [0]*node_amount
        node[(i-1)%(node_amount-1)] = 1
        node[(i+1)%(node_amount-1)] = 1
        node[-1] = 1
        adjacency_matrix.append(node)
    adjacency_matrix.append([1]*(node_amount-1) + [0])
    
    return Graph(adjacency_matrix, GraphType.WHEEL, id)

def _cycle(node_amount: int, id: str = '') -> Graph:
    """Return a cycle graph with desired amount of nodes."""
    adjacency_matrix = []
    
    for i in range(node_amount):
        n = [0]*node_amount
        n[(i-1)%node_amount] = 1
        n[(i+1)%node_amount] = 1
        adjacency_matrix.append(n)
    
    return Graph(adjacency_matrix, GraphType.CYCLE, id)

def _complete(node_amount: int, id: str = '') -> Graph:
    """Returns a complete graph with desired node amount."""
    
    adjacency_matrix = []
    
    for i in range(node_amount):
        n = [1]*node_amount
        n[i] = 0
        adjacency_matrix.append(n)
    
    return Graph(adjacency_matrix, GraphType.COMPLETE, id)


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
    """Returns all distinct ways to label n nodes of G such that none of the labeled nodes are adjacent.

    Args:
        G (Graph): Graph to use.
        n (int): Number of nodes to label.
        time_log (bool, optional): Print the runtime? Defaults to False.

    Returns:
        list: List containing all distinct labelings, each labeling has the n nodes object references to the original graph
    """

    if time_log: t = time.time()

    pairings = [[[{node},node.connections] for node in G.nodes]]

    while len(pairings) < n:
        new_pairs = []
        pair_dict = {}
        for n1 in pairings[-1]:
            for n2 in pairings[0]:
                test_set = [n1[0] | n2[0], n1[1] | n2[1]]
                if not any(val in n1[0] for val in n2[0]) and not any(val in n1[1] for val in n2[0]) and not test_set[0] in pair_dict.values():
                    new_pairs.append(test_set)
                    pair_dict[str(len(new_pairs))] = test_set[0]
        pairings.append(new_pairs)
    
    if time_log: print(f'get_nontouching_pairs runtime = {time.time()-t}secs')

    return pairings[-1]


def fast_fill(G: Graph, initial_labeling: list[Node], pinnacle_set: list[int], time_log: bool = False) -> list[Graph]:

    if time_log: t = time.time()

    G.reset_node_values()

    NP = [i for i in range(max(pinnacle_set),-1,-1) if not i in pinnacle_set]

    for i, node in enumerate(initial_labeling):
        #print(node)
        G.set_node_value(node,pinnacle_set[i])

    graph_list = [G]
    final_list = []
    total = 0
    #print(len(G.reach[0]), len(G.reach[1]))
    for val in NP:
        #print(val)
        new_graph_list = []
        for graph in graph_list:
            if val < min(pinnacle_set) and len(graph.reach[0]) + len(graph.reach[1]) == graph.size:
                total += factorial(len(graph.reach[1]))
                final_list.append(graph)
            else:
                for i, node in enumerate(graph.nodes):
                    if node in graph.reach[1]:
                        valid = True
                        if any([val > adj_node.value for adj_node in node.connections if adj_node.value in pinnacle_set]): valid = False
                        if all([val > adj_node.value for adj_node in node.connections]): valid = False
                        if valid:
                            new_graph = graph.copy()
                            new_graph.set_node_value(new_graph.nodes[i],val)
                            new_graph_list.append(new_graph)
        graph_list = new_graph_list
    
    # total += len(graph_list)
    # final_list += graph_list

    if time_log: print(f'Fast Fill runtime = {time.time()-t}secs')

    return total, final_list

def pinnaclus_utopius(G: Graph, pinnacle_set: list, time_log: bool = False) -> int | list:
    

    if time_log: t = time.time()

    not_touching = get_nontouching_nodes(G, len(pinnacle_set))

    perms = []
    _generate_permutations(perms, pinnacle_set, len(pinnacle_set))

    total = 0
    final_graph_list = []
    #print(not_touching)
    for pair in not_touching:
        for p in perms:
            tot, l = fast_fill(G,pair[0],p)
            total += tot
            final_graph_list = final_graph_list + l
    
    if time_log: print(f'Utopius runtime = {time.time()-t}secs')
    return total, final_graph_list

###########
# TESTING #
###########

G = create_graph(14,'star5')
pset = [14,13,12,11]
u = pinnaclus_utopius(G,pset,True)
print(u[0])
print(len(u[1]))


