#File to perform code refactoring before pushing the GraphUtil.py
from __future__ import annotations
from math import factorial
from dataclasses import dataclass, field
from enum import Enum, auto
import time

#################
# CUSTOM ERRORS #
#################

class ListSizeMismatch(Exception):
    """Raised when two arrays should have the same size, but do not.
    
    Attributes:
        list_one -- list in question
        list_two -- other list in question
        message -- explanation of the error
    """
    
    def __init__(self, list_one: list, list_two: list, message: str):
        self.list_one = list_one
        self.list_two = list_two
        self.message = message
        super().__init__(message)

#########
# ENUMS #
#########

class GraphType(Enum):
    STAR = auto()
    COMPLETE = auto()
    CYCLE = auto()
    WHEEL = auto()
    BIPARTITE = auto()

#################
# GRAPH CLASSES #
#################

@dataclass(eq=False)
class Node():
    value: int = field(default=0, repr=True)
    connection_list: list = field(default_factory=list, repr=False)
    
    def add_connection(self, node: Node) -> None:
        self.connection_list.append(node)
    
    def set_value(self, value: int) -> None:
        self.value = value

@dataclass
class Graph:
    id: GraphType
    adjacency_matrix: list = field(repr=False)
    size: int
    node_list: list[Node] = field(default_factory=list,repr=False)
    
    def __post_init__(self):
        for _ in range(self.size): self.node_list.append(Node(0))
        for r in range(len(self.adjacency_matrix)):
            for c in range(len(self.adjacency_matrix[r])):
                if self.adjacency_matrix[r][c]: self.node_list[r].add_connection(self.node_list[c])
    
    def set_node_values(self, values: list) -> None:
        """Assigns the values passed to the nodes in the graph"""
        
        if len(self.node_list) != len(values):
            raise ListSizeMismatch(
                list_one=self.node_list,
                list_two=values,
                message="(set_node_values) : The length of Node List does not match the length of the list of values \
                    you are attempting to assign the Node Values with."
            )
            
        for i in range(self.size):
            self.node_list[i].set_value(values[i])
    
    def get_node_values(self) -> list:
        """Returns a list of the values in the graph"""
        
        return [node.value for node in self.node_list]
    
    def get_smallest_degree(self) -> int:
        """Returns the smallest degree a Node has in this Graph"""
        
        return min([len(node.connection_list) for node in self.node_list])
    
    def get_pinnacles(self, values: list = []) -> list:
        """Returns the sorted list of pinnacle values for this graph.
        
        Attributes:
            values: list = [] -- optional argument to set the node values before getting the pinnacles
        
        Returns:
            list -- List containing the values that are pinnacles
        """
        if len(values) != 0:
            self.set_node_values(values=values)
        
        return sorted([node.value for node in self.node_list if all([node.value > other_node.value for other_node in node.connection_list])], reverse=True)

    def copy(self) -> Graph:
        """Create and return a deep copy."""
        new_graph = Graph(self.id,self.adjacency_matrix,self.size)
        new_graph.set_node_values(self.get_node_values())
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

def create_graph(node_amount: int, style: str = 'star-1') -> Graph:
    """Create a Graph with the desired amount of nodes for given style
    
    Attributes:
        node_amount: int -- total number of nodes / vertices in the Graph
        style: str -- string that determines that specific of the graph type
    """
    
    if style[0:4] == 'star':
        return _star(node_amount=node_amount, star_amount=digits_only(style))
    elif style[0:9] == 'bipartite':
        return _bipartite(node_amount=node_amount, left_amount=digits_only(style))
    elif style[0:5] == 'wheel':
        return _wheel(node_amount=node_amount)
    elif style[0:5] == 'cycle':
        return _cycle(node_amount=node_amount)
    elif style[0:8] == 'complete':
        return _complete(node_amount=node_amount)

def _star(node_amount: int, star_amount: int) -> Graph:
    """Create a Star Graph Instance"""
    
    adjacency_matrix = []
    for i in range(node_amount):
        if i < star_amount:
            node = [1]*node_amount
            node[i] = 0
        else: node = [1]*star_amount + [0]*(node_amount-star_amount)
        adjacency_matrix.append(node)
    return Graph(GraphType.STAR, adjacency_matrix, node_amount) 

def _bipartite(node_amount: int, left_amount: int):
    """Create a Complete Bipartite"""

    adjacency_matrix = []
    for i in range(left_amount):
        adjacency_matrix.append([0]*left_amount + [1]*(node_amount-left_amount))
        
    for i in range(node_amount-left_amount):
        adjacency_matrix.append([1]*left_amount + [0]*(node_amount-left_amount))
        
    return Graph(GraphType.BIPARTITE, adjacency_matrix, node_amount)

def _wheel(node_amount: int) -> Graph:
    """Returns a wheel graph with desired node amount."""
    
    adjacency_matrix = []
    
    for i in range(node_amount-1):
        node = [0]*node_amount
        node[(i-1)%(node_amount-1)] = 1
        node[(i+1)%(node_amount-1)] = 1
        node[-1] = 1
        adjacency_matrix.append(node)
    adjacency_matrix.append([1]*(node_amount-1) + [0])
    
    return Graph(GraphType.WHEEL, adjacency_matrix, node_amount)

def _cycle(node_amount: int) -> Graph:
    """Return a cycle graph with desired amount of nodes."""
    adjacency_matrix = []
    
    for i in range(node_amount):
        n = [0]*node_amount
        n[(i-1)%node_amount] = 1
        n[(i+1)%node_amount] = 1
        adjacency_matrix.append(n)
    
    return Graph(GraphType.CYCLE, adjacency_matrix, node_amount)

def _complete(node_amount: int) -> Graph:
    """Returns a complete graph with desired node amount."""
    
    adjacency_matrix = []
    
    for i in range(node_amount):
        n = [1]*node_amount
        n[i] = 0
        adjacency_matrix.append(n)
    
    return Graph(GraphType.COMPLETE, adjacency_matrix, node_amount)

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

#############################
# COMPREHENSIVE BRUTE FORCE #
#############################

def pinnaclus_brutus(G: Graph, time_log: bool = False) -> dict:
    """Generate every pinnacle set and record the number of occurences."""
    
    if time_log: t = time.time()
    pinnacle_occurences = {}
    for p in generate_permutations(G.size):
        pinn_set = G.get_pinnacles(p)
        if str(pinn_set) in pinnacle_occurences:
            pinnacle_occurences[str(pinn_set)][0] += 1
        else: pinnacle_occurences[str(pinn_set)] = [1, pinn_set]
    if time_log: print(f"pinnaclus_brutus runtime = {time.time()-t}secs")
    return pinnacle_occurences

#####################
# UTOPIUS ALGORITHM #
#####################

def get_non_touching_pairs(G: Graph, pair_size: int, time_log: bool = False) -> list:
    """Return all the distinct ways to label G with pair_size number of values such that none of them are touching."""
    if time_log: t = time.time()
    original_values = G.get_node_values()
    G.set_node_values([i+1 for i in range(G.size)])
    pairings = [[[[node], set(node.connection_list)] for node in G.node_list]]
    while len(pairings) < pair_size:
        pair_dict = {}
        pairs = []
        for n1 in pairings[-1]:
            for n2 in pairings[0]:
                if set(n1[0]) != set(n2[0]) and (not n2[0][0] in set(n1[1])) and (not n2[0][0] in set(n1[0])) and not (set(n1[0]) | set(n2[0]) in pair_dict.values()):
                    pairs.append([set(n1[0]) | set(n2[0]), n1[1] | n2[1]])
                    pair_dict[str([n1,n2])] = set(n1[0]) | set(n2[0])
        pairings.append(pairs)
    G.set_node_values(original_values)
    if time_log: print(f"get_non_touching_pairs runtime = {time.time()-t}secs")  
    return pairings[-1]

def fill_in_pair(G: Graph, pair: list, pinnacle_set_ordered: list, time_log: bool = False) -> list:
    """For a given pinnacle set with a labeling, return all of the distinct way to label the graph."""
    
    if time_log: t = time.time()
    
    G.set_node_values([0]*G.size)
    
    NP = [i for i in range(G.size,0,-1) if not i in pinnacle_set_ordered and i > G.get_smallest_degree()]
    
    for i,p in enumerate(pair[0]): p.set_value(pinnacle_set_ordered[i])
    
    graph_list = [G]
    
    for val in NP:
        new_graph_list = []
        for g in graph_list:
            for node in g.node_list:
                if node.value == 0 and all([val < other_node.value for other_node in node.connection_list if other_node.value in pinnacle_set_ordered]) and any([val < other_node.value for other_node in node.connection_list]):
                    node.set_value(val)
                    new_graph_list.append(g.copy())
                    node.set_value(0)
        graph_list = new_graph_list
    
    if time_log: print(f"fill_in_pair runtime = {time.time()-t}secs")
    
    return len(graph_list), graph_list
 
def pinnaclus_utopius(G: Graph, pinnacle_set: list, time_log: bool = False) -> int | list:
    """Return the value of Pinn(pinnacle_set, G) along with the list of all labelings minus final factorial."""
     
    if time_log: t = time.time()
     
    pairs = get_non_touching_pairs(G, len(pinnacle_set))
     
    permutations = []
    P = sorted(pinnacle_set, reverse=True)
    _generate_permutations(permutations, P, len(P))
     
    total = 0
    labelings = []
    for p in pairs:
        for i in permutations:
            pack = fill_in_pair(G,p,i)
            total += pack[0]
            labelings += pack[1]
            
    if time_log: print(f"pinnaclus_utopius runtime = {time.time()-t}secs")
    
    return factorial(G.get_smallest_degree())*total, labelings
    
#################
# TIME ANALYSIS #
#################

def time_analysis(G: Graph) -> None:
    """Prints out the pinnaclus brutus time for G, and the cumsum for utopius on all p-sets."""
    
    brute = pinnaclus_brutus(G, time_log=True)
    print(brute)
    
    psets = [p[1] for p in brute.values()]
    
    total_time = 0
    for p in psets:
        print(p)
        t = time.time()
        print(pinnaclus_utopius(G,p,time_log=True)[0])
        total_time += time.time() - t
    print(f"pinnaclus utopius cumulative runtime = {total_time}secs")


#######################
# FORMULA COMPUTATION #
#######################

class PinnacleFormulaNotDerived(Exception):
    """Raised When attempting to calculate pinn for a graph where an equation has not been derived yet."""
    
    def __init__(self,GT: GraphType):
        self.GT = GT
        self.message = f"Formula has not been derived for {GT} yet."
        super().__init__(self.message)

def pinnacle_computation(GT: GraphType, pinnacle_set: list, star_count: int = 0, bipartite_left: int = 0, time_log: bool = False) -> int:
    """Use derived formulas to do the pinnacle computation."""
    
    if time_log: t = time.time()
    
    pinnacle_set.sort(reverse=True)
    
    if GT == GraphType.COMPLETE:
        #INSERT VALIDATOR HERE
        pinn = factorial(pinnacle_set[0])
    elif GT == GraphType.STAR:
        #INSERT VALIDATOR HERE
        if len(pinnacle_set) == 1:
            pinn = star_count*(factorial(pinnacle_set[0]-1) + (pinnacle_set[0]-star_count)*factorial(pinnacle_set[0]-2))
        else:
            i = pinnacle_set[0]-pinnacle_set[-1]
            pinn = star_count * (factorial(pinnacle_set[0]-star_count)/factorial(pinnacle_set[0]-star_count-i-1)) * factorial(pinnacle_set[0]-i-2)
    elif GT == GraphType.BIPARTITE:
        #INSERT VALIDATION HERE
        m = bipartite_left
        n = pinnacle_set[0] - m
        if n > m: m, n = n, m
        i = pinnacle_set[0]-pinnacle_set[-1]
        if i < n:
            pinn = factorial(m+n-i-2)*(m*factorial(n)/factorial(n-i-1) + n*factorial(m)/factorial(m-i-1))
        else:
            pinn = factorial(m+n-i-2)*n*factorial(m)/factorial(m-i-1)
    else:
        raise PinnacleFormulaNotDerived(GT)
    
    if time_log: print(f"pinnacle_computation({GT}, {pinnacle_set}) runtime = {time.time()-t}")
    return pinn