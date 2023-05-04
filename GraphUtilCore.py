from __future__ import annotations
import time
from math import factorial
from enum import Enum, auto

###########
# CLASSES #
###########

class Node:

    def __init__(self, id: int):
        self.id = id
        self.value = 0
        self.connections = set()

    def __str__(self):
        return f'Value = {self.value}, Connections = {self.connections}'
    
    def set_value(self, value: int) -> list[int]:
        self.value = value
        return self.connections
    
    def copy(self) -> Node:
        n = Node(self.id)
        n.value = self.value
        n.connections = self.connections.copy()
        return n
    
class Graph:

    def __init__(self,matrix: list[list[int]], init: bool = True):
        self.matrix = matrix
        self.reach = set()
        self.nodes = []
        if init:
            self.nodes = [Node(_) for _ in range(len(matrix))]
            for r in range(len(matrix)):
                for c in range(len(matrix[r])):
                    if matrix[r][c]: self.nodes[r].connections.add(c)
    
    def __str__(self):
        s = 'Graph\n-----\n'
        for node in self.nodes:
            s += f'{node}\n'
        return s
    
    def set_node_value(self, index: int, value: int):
        self.reach = self.reach.union(self.nodes[index].set_value(value))
    
    def reset(self) -> None:
        for node in self.nodes:
            node.value = 0
        self.reach = set()

    def copy(self) -> Graph:
        G = Graph(self.matrix,init=False)
        for node in self.nodes:
            G.nodes.append(node.copy())
        G.reach = self.reach.copy()
        return G

    def get_smallest_degree(self):
        return min([len(self.nodes[i].connections) for i in range(len(self.nodes))])

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
    return Graph(adjacency_matrix)

def _star(node_amount: int, star_amount: int, id: str = '') -> Graph:
    """Create a Star Graph Instance"""
    
    adjacency_matrix = []
    for i in range(node_amount):
        if i < star_amount:
            node = [1]*node_amount
            node[i] = 0
        else: node = [1]*star_amount + [0]*(node_amount-star_amount)
        adjacency_matrix.append(node)
    return Graph(adjacency_matrix) 

def _bipartite(node_amount: int, left_amount: int, id: str = ''):
    """Create a Complete Bipartite"""

    adjacency_matrix = []
    for i in range(left_amount):
        adjacency_matrix.append([0]*left_amount + [1]*(node_amount-left_amount))
        
    for i in range(node_amount-left_amount):
        adjacency_matrix.append([1]*left_amount + [0]*(node_amount-left_amount))
        
    return Graph(adjacency_matrix)

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
    
    return Graph(adjacency_matrix)

def _cycle(node_amount: int, id: str = '') -> Graph:
    """Return a cycle graph with desired amount of nodes."""
    adjacency_matrix = []
    
    for i in range(node_amount):
        n = [0]*node_amount
        n[(i-1)%node_amount] = 1
        n[(i+1)%node_amount] = 1
        adjacency_matrix.append(n)
    
    return Graph(adjacency_matrix)

def _complete(node_amount: int, id: str = '') -> Graph:
    """Returns a complete graph with desired node amount."""
    
    adjacency_matrix = []
    
    for i in range(node_amount):
        n = [1]*node_amount
        n[i] = 0
        adjacency_matrix.append(n)
    
    return Graph(adjacency_matrix)

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

############
# BULKHEAD #
############

def get_nontouching_nodes(G: Graph, n: int, log: bool = False) -> list:

    if log: t = time.time()

    pairings = [[[{i}, G.nodes[i].connections] for i in range(len(G.nodes))]]

    while len(pairings) < n:
        new_pairs = []
        pair_dict = {}
        for n1 in pairings[-1]:
            for n2 in pairings[0]:
                test_set = [n1[0] | n2[0], n1[1] | n2[1]]
                if not any(val in n1[0] or val in n1[1] for val in n2[0]) and not test_set[0] in pair_dict.values():
                    new_pairs.append(test_set)
                    pair_dict[str(len(new_pairs))] = test_set[0]
        pairings.append(new_pairs)
    
    if log: print(f'get_nontouching_nodes runtime = {time.time() - t}secs')

    return pairings[-1]

def get_nontouching_vertices(G: Graph, n: int) -> list:
    if n < 1: raise ValueError
    if n == 1:
        return [[{i}, G.nodes[i].connections] for i in range(len(G.nodes))]
    L = get_nontouching_vertices(G=G, n=n-1)
    p = []
    for l in L:
        for v in range(len(G.nodes)):
            if not v in l[1] and not v in l[0]:
                l_v = [l[0] | {v}, l[1] | G.nodes[v].connections]
                if not any([l_v[0] == n[0] for n in p]):
                    p.append(l_v)
    return p

def fill_in_graph(G: Graph, pairing: list, pinnacle_set: list, log: bool = False) -> int | list:

    if log: t = time.time()

    for i, n in enumerate(pairing[0]):
        G.nodes[n].value = pinnacle_set[i]

    G.reach = pairing[0].union(pairing[1])

    NP = [i for i in range(len(G.nodes), 0, -1) if not i in pinnacle_set and i >= G.get_smallest_degree()]

    graph_list = [G]
    final_list = []
    total = 0

    #print(G)

    for val in NP:
        new_graph_list = []
        for graph in graph_list:
            for i in range(len(graph.nodes)):
                if graph.nodes[i].value == 0 and all([val < graph.nodes[adj].value for adj in graph.nodes[i].connections if graph.nodes[adj].value in pinnacle_set]) and any([val < graph.nodes[adj].value for adj in graph.nodes[i].connections]):
                    new_graph = graph.copy()
                    #print(new_graph)
                    new_graph.set_node_value(i,val)
                    if len(new_graph.reach) == len(new_graph.nodes) and val < min(pinnacle_set):
                        total += factorial(sum([1 for node in new_graph.nodes if node.value == 0]))
                        final_list.append(new_graph)
                    else: new_graph_list.append(new_graph)
        graph_list = new_graph_list
    
    if log: print(f'fill_in_graph runtime = {time.time()-t}secs')

    return total, final_list

def pinnaclus_utopius(G: Graph, pinnacle_set: list, log: bool = False) -> int | list:

    if log: t = time.time()

    notouch = get_nontouching_nodes(G, len(pinnacle_set))
    #if log: print(f'nontouching runtime = {time.time()-t}secs')

    perms = []
    _generate_permutations(perms, pinnacle_set, len(pinnacle_set))
    #if log: print(f'permuatation generation runtime = {time.time()-t}secs')

    total = 0
    final_graph_list = []

    for pair in notouch:
        for p in perms:
            tot, l = fill_in_graph(G, pair, p)
            total += tot
            final_graph_list.extend(l)
            G.reset()

    if log: print(f'Utopius Core runtime = {time.time()-t}secs')

    return total, final_graph_list

################
# COMPUTATIONS #
################

def partitions(n: int):
    if n == 0:
        yield []
        return
    
    for p in partitions(n-1):
        yield [1]+p
        if p and (len(p) < 2 or p[1] > p[0]):
            yield [p[0] + 1] + p[1:]

def composition_size(n: int, m: int):
    parts = []
    for i in partitions(n):
        if len(i) == m: parts.append(sorted(i,reverse=True))
    all_parts = []
    for p in parts:
        perms = []
        unique = {}
        _generate_permutations(perms, p, len(p))
        for i in perms:
            if not str(i) in unique.values():
                all_parts.append(i.copy())
                unique[str(len(unique))] = str(i)
    return all_parts
    
    return parts

def multinomial(n: int, l: list) -> int:
    if sum(l) != n: raise ValueError()
    total = factorial(n)
    for v in l:
        total /= factorial(v)
    return int(total)

class PinnacleFormulaNotDerived(Exception):
    """Raised When attempting to calculate pinn for a graph where an equation has not been derived yet."""
    
    def __init__(self,GT: GraphType):
        self.GT = GT
        self.message = f"Formula has not been derived for {GT} yet."
        super().__init__(self.message)

class InvalidPinnacleSet(Exception):
    """Raised when trying passing an inadmissible pinnacle set."""
    
    def __init__(self, pinnacle_set: list, message: str):
        self.pinnacle_set = pinnacle_set
        self.message = message
        super().__init__(message)

def _validate_pinnacle_set(GT: GraphType, pinnacle_set: list, node_count: int, star_count: int = 0, bipartite_left: int = 0) -> None:
    """Returns true if this is a valid pinnacle set for provide graph type."""
    
    p = sorted(pinnacle_set, reverse=True)
    
    if GT == GraphType.STAR:
        if not all([i > star_count for i in p]): raise InvalidPinnacleSet(p, f'{p} is an invalid pinnacle set for a star graph with {node_count} vertices and {star_count} stars.')
        if not node_count in p: raise InvalidPinnacleSet(p, f'{p} is an invalid pinnacle set for a star graph with {node_count} vertices and {star_count} stars.')
        for i in range(len(p)-1):
            if p[i]-p[i+1] != 1: raise InvalidPinnacleSet(p, f'{p} is an invalid pinnacle set for a star graph with {node_count} vertices and {star_count} stars.')
        return
    elif GT == GraphType.BIPARTITE:
        if not node_count in p: raise InvalidPinnacleSet(p, f"{p} is an invalid pinnacle set for a complete bipartite with {node_count} nodes and m={bipartite_left}, n={node_count-bipartite_left}.")
        min_val = min(bipartite_left, node_count-bipartite_left)
        if not all([i > min_val for i in p]): raise InvalidPinnacleSet(p, f"{p} is an invalid pinnacle set for a complete bipartite with {node_count} nodes and m={bipartite_left}, n={node_count-bipartite_left}.")
        for i in range(len(p)-1):
            if p[i]-p[i+1] != 1: raise InvalidPinnacleSet(p, f"{p} is an invalid pinnacle set for a complete bipartite with {node_count} nodes and m={bipartite_left}, n={node_count-bipartite_left}.")
        return
    elif GT == GraphType.CYCLE:
        if p[0] != node_count: raise InvalidPinnacleSet(pinnacle_set,f'{p} must contain the largest possible value (i.e., node_count).')
        for i in range(len(p)-1):
            if p[i]-p[i+1] != 1: raise InvalidPinnacleSet(p, f'{p} is a non-interval pinnacle set, thus I do not have a formulation for it yet.')
        return 
    elif GT == GraphType.WHEEL:
        if len(p) != 1: raise InvalidPinnacleSet(pinnacle_set, f'{p} is not a pinnacle set I have a formula for, for a wheel graph.')
        if p[0] != node_count: raise InvalidPinnacleSet(pinnacle_set, f'{p} must contain the node_count as a value.')
        return

def pinnacle_computation(GT: GraphType, pinnacle_set: list, node_count: int, star_count: int = 0, bipartite_left: int = 0, time_log: bool = False) -> int:
    """Use derived formulas to do the pinnacle computation."""
    
    _validate_pinnacle_set(GT, pinnacle_set, node_count, star_count, bipartite_left)
    
    if time_log: t = time.time()
    
    pinnacle_set.sort(reverse=True)
    
    if GT == GraphType.COMPLETE:
        pinn = factorial(pinnacle_set[0])
    elif GT == GraphType.STAR:
        if len(pinnacle_set) == 1:
            pinn = star_count*(factorial(pinnacle_set[0]-1) + (pinnacle_set[0]-star_count)*factorial(pinnacle_set[0]-2))
        else:
            i = pinnacle_set[0]-pinnacle_set[-1]
            pinn = star_count * (factorial(pinnacle_set[0]-star_count)/factorial(pinnacle_set[0]-star_count-i-1)) * factorial(pinnacle_set[0]-i-2)
    elif GT == GraphType.BIPARTITE:
        m = bipartite_left
        n = pinnacle_set[0] - m
        if n > m: m, n = n, m
        i = pinnacle_set[0]-pinnacle_set[-1]
        if i < n:
            pinn = factorial(m+n-i-2)*(m*factorial(n)/factorial(n-i-1) + n*factorial(m)/factorial(m-i-1))
        else:
            pinn = factorial(m+n-i-2)*n*factorial(m)/factorial(m-i-1)
    elif GT == GraphType.CYCLE:
        if len(pinnacle_set) == 0:
            pinn = node_count*2**(node_count-2)
        else:
            pinn = 0
            for part in composition_size(node_count-len(pinnacle_set),len(pinnacle_set)):
                pinn += multinomial(node_count-len(pinnacle_set),part)
            pinn = pinn * factorial(len(pinnacle_set)-1) * 2**(node_count - 2*len(pinnacle_set)) * node_count
    elif GT == GraphType.WHEEL:
        n = node_count
        pinn = factorial(n-1) + (n-1) * (2**(n-2) + sum([2**(i-2)*factorial(n-i) for i in range(2,n-1)]))
    else:
        raise PinnacleFormulaNotDerived(GT)
    
    if time_log: print(f"pinnacle_computation({GT}, {pinnacle_set}) runtime = {time.time()-t}")

    return pinn

def main():
    return

if __name__ == '__main__':
    main()