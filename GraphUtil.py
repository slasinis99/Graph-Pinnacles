from __future__ import annotations
import time
from math import factorial

"""Graph Utility for Pinnacle Sets
@author: Stephen Lasinis

This file contain the methods and classes required to perform the
computations related to calculating the Pinnacle Sets of graphs.

Classes
-------
Node(value: int = 0):
    Node object will store the value at aformentioned
    vertex as well as a list of the nodes it shares an 
    edge with.
Graph(adjacency_matrix):
    Graph object requires an adjacency matrix and will
    construct the desired Graph object from that.

Methods
-------
generatePermutations(size: int) -> list:
    returns a list of all permutations of the given size
_generatePermuations(S: list, a: list, size: int):
    recursive implementation of Heap's Algorithm
    DO NOT MANUALLY CALL THIS FUNCTION!!!
calculateStarPinn(k: int, n: int, i: int):
    Returns the number of ways the this pinnacle set can be
    created on a star graph with k-stars and n-nodes.
pinnaclusBrutus(G: Graph) -> dict:
    Brute force algorithm to determine all unique pinnacle sets on G
    as well as the number of occurences for each one.
"""

class Node:
    """
    A class that is used to represent a node / vertex
    within a graph. To be used directly in conjuction
    with the Graph class.
    
    ...
    
    Attributes
    ----------
    _value : int
        an integer representing the value of this Node
    _connectionList : list
        a list holding the Node objects this Node shares an edge with

    Methods
    -------
    getValue()
        returns the value stored at this Node
    setValue(value: int)
        assigns the value passed to the _value attribute
    getConnectionAmount()
        returns the number of edges this Node shares with other Node objects
    getConnection(index: int)
        returns the Node object at the passed index from the connections
    addConnection(node: Node)
        adds the passed Node to the list of connections
    """
    def __init__(self, value=0):
        self._value = value
        self._connectionList = []
        
    def addConnection(self, node: Node):
        self._connectionList.append(node)
    
    def setValue(self, value: int):
        self._value = value
    
    def getValue(self): 
        return self._value
    
    def getConnectionAmount(self):
        return len(self._connectionList)
    
    def getConnection(self, index):
        return self._connectionList[index]
    
    def getConnectionList(self):
        return self._connectionList
        
class Graph:
    """
    A class to be used to represent a graph or network.
    Makes use of the Node class.
    
    ...
    
    Attributes
    ----------
    _size : int
        number of nodes/vertices in our graph
    _nodeList : list
        contains the Nodes in the graph
    
    Methods
    -------
    getSize()
        returns the number of nodes in the graph
    getNodeList()
        returns the list of nodes
    getNode(index: int)
        return the node at requested index
    setNodeValues(values: list)
        sets the values at all nodes in accordance with values list
    getPinnacles(values: list = [])
        return the set of pinnacles for the graph with values
    """
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self._size = len(adjacency_matrix)
        self._nodeList = []
        for _ in range(self._size): self._nodeList.append(Node())
        for r in range(len(adjacency_matrix)):
            for c in range(len(adjacency_matrix[r])): 
                if adjacency_matrix[r][c]: self.getNode(r).addConnection(self.getNode(c))

    def getNodeList(self):
        return self._nodeList
    
    def getNode(self, index: int):
        return self._nodeList[index]
    
    def getSize(self):
        return self._size
    
    def setNodeValues(self, values: list):
        for i in range(self._size):
            self._nodeList[i].setValue(values[i])
    
    def getPinnacles(self, values: list = []):
        #If values length is nonzero, then assign the values
        if len(values) == self.getSize():
            self.setNodeValues(values)
        
        pinnacles = []
        #Loop through the nodes and determine if each is a pinnacle
        for node in self.getNodeList():
            pinn = True
            for o in node.getConnectionList():
                if node.getValue() < o.getValue(): pinn = False
            if pinn:
                pinnacles.append(node.getValue())
        pinnacles.sort(reverse=True)
        return pinnacles
    
    def copy(self)->Graph:
        #Create a new graph with our adjacency matrix, and set the values of all the nodes to our values
        newG = Graph(self.adjacency_matrix)
        
        #Get the list of nodes
        newNodes = newG.getNodeList()
        oldNodes = self.getNodeList()
        
        #Loop through copying over the values
        for i in range(len(newNodes)):
            newNodes[i].setValue(oldNodes[i].getValue())
        
        #Return the new graph
        return newG
    
    def getSmallestDegree(self)->int:
        #Loop through the nodes
        val = self._size
        for node in self.getNodeList():
            if len(node.getConnectionList()) < val: val = len(node.getConnectionList())
        return val
        

#Inputs : integer n for the number of nodes, style string
def createGraph(n: int, style='star-1') -> Graph:
    """
    Method that will create a Graph object of a desired type for you.

    Args:
        n (int): number of total nodes
        style (str, optional): the style of graph you want
            Style Formats:
                - star-"number of stars here"
                - wheel
                -bipartite "number of left nodes"-"number of right nodes"

    Returns:
        Graph: 
    """
    #Handle which style is being created
    if style[0:4] == 'star':
        N = ''
        for char in style:
            if char.isnumeric(): N += char
        N = int(N)
        
        #Create the adjacency matrix
        adjacency_matrix = []
        
        #Create then nodes
        for i in range(n):
            #Declare an empty node
            if i < N:
                node = [1] * n 
                node[i] = 0
            else:
                node = [1] * N + [0] * (n-N)
            
            #Append this node to the adjacency matrix
            adjacency_matrix.append(node)
    #Handle the wheel graph
    if style[0:5] == 'wheel':
        #Create the adjacency matrix
        adjacency_matrix = []
        
        #Loop to create the nodes
        for i in range(n-1):
            node = [0] * n
            node[(i-1)%(n-1)] = 1
            node[(i+1)%(n-1)] = 1
            node[n-1] = 1
            adjacency_matrix.append(node)
        node = [1]*n
        node[n-1] = 0
        adjacency_matrix.append(node)
    #Handle the basic cycle graph
    if style[0:5] == 'cycle':
        #Create the adjacency matrix
        adjacency_matrix = []
        
        #Loop to create the nodes
        for i in range(n):
            node = [0] * n
            node[(i-1)%n] = 1
            node[(i+1)%n] = 1
            adjacency_matrix.append(node)
    #Handle a complete bipartite graph
    if style[0:9] == 'bipartite':
        #Adjacency Matrix
        adjacency_matrix = []
        #Parse m
        m = ''
        for i in range(len(style)):
            if style[i].isnumeric(): 
                m += style[i]
            else:
                if len(m) != 0:
                    style = style[i+1:]
                    break
        m = int(m)
        #Parse n
        n = ''
        for i in range(len(style)):
            if style[i].isnumeric(): n += style[i]
        n = int(n)
        #Loop through the nodes to create
        for i in range(m):
            node = [0]*m + [1]*n
            adjacency_matrix.append(node)
        for i in range(n):
            node = [1]*m + [0]*n
            adjacency_matrix.append(node)
        
    #Create the Graph Object and return it
    return Graph(adjacency_matrix)

#Parent function to the recursive method for
#generating all permutations given a size
def generatePermutations(size: int) -> list:
    S = [] 
    a = []
    for i in range(size): a.append(i+1)
    _generatePermutations(S, a, size)
    return S  

#Useful Recursive Algorithm to Generate the Symmetric Group of Permutations
#Never manually call this method, always call the parent 
#(unless you're me and you know what the fuck you're doing)
def _generatePermutations(S: list, a: list, size: int):
    #If size is 1, then store permutation in S
    if (size == 1):
        S.append(a.copy())
        #print(a)
        return
    
    #Loop through our sizing
    for i in range(size):
        _generatePermutations(S, a, size-1)
        
        #If size is odd, swap the first and last element
        #else swap the ith and last element
        if (size & 1):
            a[0], a[size-1] = a[size-1], a[0]
        else:
            a[i], a[size-1] = a[size-1], a[i]

#Function for calculating Pinn(P) for P={ n, n-1, ..., n-i } 
# for a k-star graph with n vetices
def calculateStarPinn(k: int, n: int, i: int) -> int:
    if i == 0:
        return int(k*factorial(n-1) + k*(n-k)*factorial(n-2))
    elif i < n-k:
        return int(k * (factorial(n-k) / factorial(n-i-k-1)) * factorial(n-i-2))
    else: return 0

#Function for calculating Pinn(P) for P={ n, n-1, ..., n-i } 
# for a Bipartite Graph with m left nodes and n right nodes
def calculateBipartitePinn(m: int, n: int, i: int)->int:
    if m < n: m, n = n, m
    if i < n:
        return int(factorial(m+n-i-2)*(m * (factorial(n) / factorial(n-i-1)) + n * (factorial(m) / factorial(m-i-1))))
    elif i < m:
        return int(factorial(m+n-i-2)*(n * factorial(m) / factorial(m-i-1)))
    else: return 0

#####################
# ALGORITHM CENTRAL #
#####################
def pinnaclusBrutus(G: Graph) -> dict:
    """
    Algorithm that will systematically compute the pinnacles
    for every possible permutation of the values in G.

    Args:
        G (Graph): Graph object to perform the method on.

    Returns:
        dict: Dictionary containing the unique pinnacle sets and the
        number of occurences of each
    """
    #Log the time to completion
    t = time.time()
    #Variable to Keep Track of the occurences of each pinnacle set
    pinnacle_occurences = {}
    #Generate all Permutations for the size of G
    pList = generatePermutations(G.getSize())
    
    #Loop through the permutations and assign them to G
    for p in pList:
        #Get the pinnacles for this permutation
        pinnacles = G.getPinnacles(p)
        pinnacles = sorted(pinnacles, reverse=True)
        #Increment occurences
        if str(pinnacles) in pinnacle_occurences:
            pinnacle_occurences[str(pinnacles)] += 1
        else: pinnacle_occurences[str(pinnacles)] = 1
    
    #Print the runtime
    print(f"pinnaclusBrutus runtime : {time.time()-t}sec")
    
    #Return the occurence dictionary
    return pinnacle_occurences

def pinnaclusPairalingus(G: Graph, count: int) -> list:
    """Generate All Node Combos s.t. no 2 are touching.

    Args:
        G (Graph): Graph object in question
        count (int): number of nodes to not be touching
    
    Returns:
        list: a list of all unique node combinations
    """
    #Keep a time log
    t = time.time()
    #Keep track of all unique pairings
    pairing_dict = {}
    #Keep track of the possible pairings
    pairings = []
    
    #Create the first set of pairings (i.e., each node)
    pairs = []
    for i in G.getNodeList():
        #For each node create a set of itself, and a set of its connections
        pairs.append([set([i]), set(i.getConnectionList())])
    pairings.append(pairs) 
    
    #Create new pairings until the length of pairings == count
    while(len(pairings) < count):
        pairs = []
        #Loop through the last row in pairings
        for i in range(len(pairings[len(pairings)-1])):
            #Loop through the first row in pairings
            for j in G.getNodeList():
                #Check if this node is not in i's set
                if not j in pairings[len(pairings)-1][i][1] and not j in pairings[len(pairings)-1][i][0]:
                    #Make sure this pairing is not in the pairing_dict
                    if not set(pairings[len(pairings)-1][i][0] | set([j])) in pairing_dict.values():
                        pairs.append([pairings[len(pairings)-1][i][0] | set([j]), pairings[len(pairings)-1][i][1] | set(j.getConnectionList())])
                        pairing_dict[str(pairings[len(pairings)-1][i][0] | set([j]))] = pairings[len(pairings)-1][i][0] | set([j])
        pairings.append(pairs)
    #print(f"Number of ways to label {count} vertice(s) such that none are touching:\n\t{len(pairings[len(pairings)-1])}")
    print(f"pinnaclusPairalingus Runtime : {time.time()-t}")
    return pairings[len(pairings)-1]

def pinnaclusUtopius(P: list, G: Graph):
    """Algorithm to generate number of labelings for P on G

    Args:
        P (pinnacle set list): List containing the desired pinnacle values
        G (Graph in question): Graph object corresponding to the graph you want to check

    Returns:
        list, float: returns the list of valid graphs filled until the remaining labelings are factorial, also returns runtime
    """
    #Log the Runtime
    t = time.time()
    
    #Sort P into descending order
    P.sort(reverse=True)
    
    #Create a list of the values not in P
    NP = []
    for i in range(P[0],0,-1):
        if not i in P: NP.append(i)
    
    #Get the list of all distinct pairs of where the values of P can go
    pairs = pinnaclusPairalingus(G, len(P))
    perms = []
    _generatePermutations(perms, P.copy(), len(P))
    
    #Create the starting list of possible graphs
    graph_list = []
    #Loop through all the pairs
    for pair in pairs:
        pl = list(pair[0])
        #Now loop through the permutations
        for p in perms:
            for i in range(len(pl)):
                pl[i].setValue(p[i])
            #Now copy this into a new graph
            newG = G.copy()
            graph_list.append(newG)
            #Now reset the values to zero
            for i in range(len(pl)):
                pl[i].setValue(0)
    #print(graph_list)
    #print(f"Graph List Length = {len(graph_list)}")
    
    #Loop through the NP List
    for i in NP:
        if i > G.getSmallestDegree():
            #Declare list to store new graphs
            new_graphs = []
                
            #For each of these i values, look through the graphs in graph list
            for g in graph_list:
                #print(f"Current Graph = {g}")
                #For this graph, loop through the node list
                nl = g.getNodeList()
                
                for j in range(len(nl)):
                    #Check that this node has value 0
                    #print(f"Current node value = {nl[j].getValue()}")
                    if nl[j].getValue() == 0:
                        #Go through the connections and make sure that at least one is larger than us
                        #and make sure if we are touching a value in P, that we are smaller
                        valid = True
                        higher = False
                        allZero = True
                        nc = nl[j].getConnectionList()
                        for c in range(len(nc)):
                            #First check if the value of node c is in P
                            if nc[c].getValue() in P:
                                #If i is larger than nc[c] then we are invalid
                                #print(f"i={i} vs p-node={nc[c].getValue()}")
                                if i > nc[c].getValue(): valid = False
                            if i < nc[c].getValue(): higher = True
                        #If higher and valid, modify our graph, copy it, then return the original back to how it was
                        if higher == True and valid == True:
                            nl[j].setValue(i)
                            newG = g.copy()
                            new_graphs.append(newG)
                            nl[j].setValue(0)
                            #print(f"New Graph Created : {newG}")
            graph_list = new_graphs
    #print(f"Number of Graphs = {len(graph_list) * factorial(g.getSmallestDegree())}")
    #print(f"Pinnaclus Utopius Runtime : {time.time()-t}sec")
    return len(graph_list) * factorial(G.getSmallestDegree()), graph_list, time.time()-t

####################
#   VERIFICATION   #
####################

def verifyBipartite(m,n) -> None:
    print(f"Verification for Bipartite {m}-{n}")
    t = time.time()
    for i in range(m):
        #Generate our P set
        P = []
        for v in range(m+n,m+n-i-1,-1):
            P.append(v)
        
        #Print the current complete P set
        print(f"For P = {P}")
        
        #Get the value from pinnaclus utopius and the calculated count
        actual = pinnaclusUtopius(P, createGraph(m+n,style=f"bipartite {m}-{n}"))[0]
        calc = calculateBipartitePinn(m,n,i)
        
        #Print the results
        print(f"\tActual = {actual}")
        print(f"\tCalc   = {calc}")
    print(f"Verify Bipartite Runtime : {time.time() - t}sec")
