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
    CUSTOM = auto()

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

def create_graph_custom(adjacency_matrix: list):
    return Graph(GraphType.CUSTOM,adjacency_matrix,len(adjacency_matrix))

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
    """Return all the distinct ways to label G with pair_size number of values such that none of them are touching.

    Args:
        G (Graph): Graph to Use
        pair_size (int): number of nodes to label
        time_log (bool, optional): Do you want it to display the runtime?. Defaults to False.

    Returns:
        list: returns a list of the distinct nodes to label such that none are touching
            : Format = [[...,{labeled nodes that do not touch}, {nodes that we are adjacent to},...]]
    """

    #Log the time
    if time_log: t = time.time()

    #Store the original values in the Graph just in case
    original_values = G.get_node_values()

    #Label all the Nodes to more easily distinguish them
    G.set_node_values([i+1 for i in range(G.size)])

    #Create the first list of pairings (i.e., 1 node labeled, just the list of nodes)
    pairings = [[[[node], set(node.connection_list)] for node in G.node_list]]

    #Loop until we have the pairings up to the desired node count
    while len(pairings) < pair_size:
        #Keep track of the new pairs as well as a dictionary to avoid duplicates
        pair_dict = {}
        pairs = []

        #Loop through the most recent set of pairs as well as the individual nodes to see
        # which can be added such that none are touching
        for n1 in pairings[-1]:
            for n2 in pairings[0]:
                #Make sure the two sets are not equal, and that the new node is not in our pair or touching our pair
                # and that it is distinct from all other pairs of this size
                if set(n1[0]) != set(n2[0]) and (not n2[0][0] in set(n1[1])) and (not n2[0][0] in set(n1[0])) and not (set(n1[0]) | set(n2[0]) in pair_dict.values()):
                    #Append the set of nodes in the pair as well as the nodes they are adjacent to
                    pairs.append([set(n1[0]) | set(n2[0]), n1[1] | n2[1]])
                    #Update the dictionary
                    pair_dict[str([n1,n2])] = set(n1[0]) | set(n2[0])
        #Append to the cumulative list of pairings
        pairings.append(pairs)
    
    #Reset the node values to their original
    G.set_node_values(original_values)

    #Print the runtime if requested
    if time_log: print(f"get_non_touching_pairs runtime = {time.time()-t}secs")  

    #Return the final list in pairings
    return pairings[-1]

def fill_in_pair(G: Graph, pair: list[Node], pinnacle_set_ordered: list, time_log: bool = False, complete: bool = False) -> list:
    """For a given pinnacle set with a labeling, return all of the distinct way to label the graph.

    Args:
        G (Graph): Graph to use.
        pair (list): list of nodes that should be labeled with pinnacle set
        pinnacle_set_ordered (list): list of the pinnacle values in descending order
        time_log (bool, optional): Do you want to print the runtime? Defaults to False.
        complete (bool, optional): Should we fill in the graphs completely? (Takes longer) Defaults to False.

    Returns:
        list: A list of "all" Graph objects with the proper labelings.
    """
    #Log the time if requested
    if time_log: t = time.time()
    
    #Set the node values to all 0's
    G.set_node_values([0]*G.size)
    
    #If we want the graphs to be complete
    if complete == True:
        #NP are all other values that are not pinnacles
        NP = [i for i in range(G.size,0,-1) if not i in pinnacle_set_ordered]
    else: 
        #If not, NP contains the values down to 1 + the smallest degree
        NP = [i for i in range(G.size,0,-1) if not i in pinnacle_set_ordered and i > G.get_smallest_degree()]
    
    #Set the appropriate nodes to our pinnacle values
    for i,p in enumerate(pair[0]): p.set_value(pinnacle_set_ordered[i])
    
    #Begin our comprehensive list of graphs with the starting Graph
    graph_list = [G]
    
    #Loop through the values in NP (descending order)
    for val in NP:
        #Create a placeholder for the new graphs
        new_graph_list = []

        #Loop through each graph in graph_list, and all nodes within those graphs
        for g in graph_list:
            for node in g.node_list:
                #Check if we can place our NP value at this node
                if node.value == 0 and all([val < other_node.value for other_node in node.connection_list if other_node.value in pinnacle_set_ordered]) and any([val < other_node.value for other_node in node.connection_list]):
                    #Yes? Then set the value and append a copy of this graph
                    node.set_value(val)
                    new_graph_list.append(g.copy())
                    #Then reset the value so it does not interfere later
                    node.set_value(0)
        #Override the old graph list with our new graphs
        graph_list = new_graph_list
    
    #Print runtime if requested.
    if time_log: print(f"fill_in_pair runtime = {time.time()-t}secs")
    
    #Return the length of the graph list as well as the list itself.
    return len(graph_list), graph_list
 
def pinnaclus_utopius(G: Graph, pinnacle_set: list, time_log: bool = False, complete: bool = False) -> int | list:
    """Return the value of Pinn(pinnacle_set, G) along with the list of all labelings minus final factorial.

    Args:
        G (Graph): Graph to Use
        pinnacle_set (list): list of pinnacles to use
        time_log (bool, optional): Print runtime? Defaults to False.
        complete (bool, optional): Generate all graph possibilities? (Slow.) Defaults to False.

    Returns:
        int | list: returns the total number of possible labelings as well as the list of them
    """
    
    #Log the time if requested.
    if time_log: t = time.time()
    
    #Get all the distinct pairs of where we can place
    # the pinnacles
    pairs = get_non_touching_pairs(G, len(pinnacle_set))
    
    #Generate the list of permutations for the pinnacle set
    permutations = []
    P = sorted(pinnacle_set, reverse=True)
    _generate_permutations(permutations, P, len(P))
    
    #Keep track of the total number of labelings
    total = 0

    #Also declare the list to store the graphs in
    labelings = []

    #Loop through all the pairs as well as all unique permutations
    # of the pinnacle set
    for p in pairs:
        for i in permutations:
            #Fill in the graph with this pinnacle set labeling
            pack = fill_in_pair(G,p,i,complete=complete)

            #Increment the total, and append the graphs to our list
            total += pack[0]
            labelings += pack[1]

    #Print runtime if requested.       
    if time_log: print(f"pinnaclus_utopius runtime = {time.time()-t}secs")
    
    #If we did not generate all labelings, then we must account
    # for the discrepency in our total
    if not complete:
        total = int(factorial(G.get_smallest_degree())*total)
    
    #Return the number of labelings and the labelings we generated.
    return total, labelings
    
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
        if len(p) != 1: raise InvalidPinnacleSet(pinnacle_set, f'{p} is not a pinnacle set I have a formula for, for a cycle graph.')
        if p[0] != node_count: raise InvalidPinnacleSet(pinnacle_set,f'{p} must contain the largest possible value (i.e., node_count).')
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
        return node_count*2**(node_count-2)
    elif GT == GraphType.WHEEL:
        n = node_count
        pinn = factorial(n-1) + (n-1) * (2**(n-2) + sum([2**(i-2)*factorial(n-i) for i in range(2,n-1)]))
    else:
        raise PinnacleFormulaNotDerived(GT)
    
    if time_log: print(f"pinnacle_computation({GT}, {pinnacle_set}) runtime = {time.time()-t}")

    return pinn

###############################
# PINNACLE SET ANALYSIS STUFF #
###############################

def analyis_interval(G: Graph) -> None:
    D = pinnaclus_brutus(G)
    
    #Find the total number of labelings that are intervals
    total = sum([s[0] for s in D.values() if all([s[1][i]-s[1][i+1] == 1 for i in range(0,len(s[1])-1)])])
    print(f"Number of labelings for interval pinnacle sets = {total}")
    
    #find the total number of labelings that are not intervals
    total = sum([s[0] for s in D.values() if any([s[1][i]-s[1][i+1] != 1 for i in range(0,len(s[1])-1)])])
    print(f"Number of labelings for non-interval pinnacle sets = {total}")
    
###############################
# UNIFORM PINNACLE SET SEARCH #
###############################

def increment_matrix(adjacency_matrix: list, key: list = [0,0]):
    if key[0] == key[1]:
        if key[1]+1 < len(adjacency_matrix): increment_matrix(adjacency_matrix,[key[0],key[1]+1])
        else:
            if key[0]+1 < len(adjacency_matrix): increment_matrix(adjacency_matrix,[key[0]+1,key[0]+1])
            else: return
    else:
        if adjacency_matrix[key[0]][key[1]] == 0: 
            adjacency_matrix[key[0]][key[1]] = 1
            adjacency_matrix[key[1]][key[0]] = 1
            return
        else:
            adjacency_matrix[key[0]][key[1]] = 0
            adjacency_matrix[key[1]][key[0]] = 0
            if key[1] < len(adjacency_matrix)-1: increment_matrix(adjacency_matrix,[key[0],key[1]+1])
            else:
                if key[0] < len(adjacency_matrix)-1: increment_matrix(adjacency_matrix,[key[0]+1,key[0]+1])
                else: return

def uniform_search(adjacency_matrix: list):
    uniform = []
    increment_matrix(adjacency_matrix)
    while adjacency_matrix != [[0]*len(adjacency_matrix)]*len(adjacency_matrix):
        #print(adjacency_matrix)
        if all([1 in row for row in adjacency_matrix]):
            p = pinnaclus_brutus(create_graph_custom(adjacency_matrix))
            vals = list(p.values())
            if all(val[0] == vals[0][0] for val in vals):
                print(p)
                print(adjacency_matrix)
                new = []
                for r in adjacency_matrix: new.append(r.copy())
                uniform.append(new)
        increment_matrix(adjacency_matrix)
    #print(uniform)
    print(f'Done')
    
# adj = []
# for i in range(6):
#     adj.append([0]*6)

# t = time.time()
# uniform_search(adj)
# print(f'Total Time = {time.time()-t}secs')

G = create_graph(20,'wheel')
print(len(get_non_touching_pairs(G,9,True)))