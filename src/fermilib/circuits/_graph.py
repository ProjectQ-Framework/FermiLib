#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import

import queue


class Node:
    """Graph node

    These graph nodes may be used to store data or other attributes within
    the Graph structure.
    """
    def __init__(self, value=None):
        """Build a graph node initialized with generic value"""
        self.value = value


class Graph:
    """An undirected graph of nodes of arbitrary connectivity

    A generic graph class for undirected graphs that holds Nodes and
    edges that connect them.

    Attributes:
        nodes(list): A list of Node objects containing the nodes of the graph
        node_uids(list): A list of unique IDs (uids) for each node
        uid_to_index(dict): A dictionary that maps UIDs currently present
            in the graph to their node index
        neigbors(list of sets):  A list of sets that enumerate the neighbors
            of each node.  For example the neighbors of node i are the set
            neighbors[i]
        next_uid(int):  The next unique ID to be assigned to a node on addition
    """

    def __init__(self):
        """Set up an empty graph with no nodes"""
        self.nodes = []
        self.node_uids = []
        self.uid_to_index = {}
        self.neighbors = []
        self.next_uid = 0

    def add_node(self, node=Node()):
        """Add a Node to the graph.

        Args:
            node(Node): A Node object to be added to the graph

        Returns(int): The unique identified that was given to the node
        """
        self.nodes += [node]
        self.node_uids += [self.next_uid]
        self.neighbors += [set()]
        self.uid_to_index[self.next_uid] = self.node_count() - 1
        self.next_uid += 1

    def remove_node(self, node_id):
        """Remove a graph node

        This removes a node from the graph and leverages the unique ID
        system used internally to avoid having to modify the edges for all
        nodes in the graph.

        Args:
            node_id(int): Index of the node to be removed.
        """
        if node_id >= self.node_count():
            raise IndexError("Node ID out of range for graph")
        node_uid = self.node_uids[node_id]
        for i in range(self.node_count()):
            if node_uid in self.neighbors[i]:
                self.neighbors[i].remove(node_uid)
        for i in range(node_id + 1, self.node_count()):
            self.uid_to_index[self.node_uids[i]] -= 1

        del self.uid_to_index[node_uid]
        del self.nodes[node_id]
        del self.neighbors[node_id]
        del self.node_uids[node_id]

    def node_count(self):
        """Number of nodes in the graph

        Returns(int): Number of nodes currently in the graph
        """
        return len(self.nodes)

    def add_edge(self, node_id1, node_id2):
        """Add an edge between node 1 and node2

        Args:
            node_id1(int): Index of first node on edge
            node_id2(int): Index of second node on edge
        """
        if node_id1 >= self.node_count() or node_id2 >= self.node_count():
            raise IndexError("Node ID out of range for graph.")
        if node_id1 == node_id2:
            raise IndexError("Error, self-loops not supported.")

        self.neighbors[node_id1].add(self.node_uids[node_id2])
        self.neighbors[node_id2].add(self.node_uids[node_id1])

    def remove_edge(self, node_id1, node_id2):
        """Remove an edge between node1 and node2

        Args:
            node_id1(int): Index of first node on edge
            node_id2(int): Index of second node on edge
        """
        if node_id1 >= self.node_count() or node_id2 >= self.node_count():
            raise IndexError("Node ID out of range for graph")

        self.neighbors[node_id1].remove(self.node_uids[node_id2])
        self.neighbors[node_id2].remove(self.node_uids[node_id1])

    def find_index(self, value, starting_node=0):
        """Find the index of the first node that matches value in a BFS

        Performs a breadth-first search of the graph starting at node index
        starting_node.  Returns the index or None if no match is found

        Args:
            value(Node Value) - Value to match against in the graph
            starting_node(int) - Node index to start search from
        """
        if starting_node > self.node_count():
            raise IndexError("Node ID out of range.")

        node_queue = queue.LifoQueue()
        node_queue.put(starting_node)
        visited = [starting_node]
        while not node_queue.empty():
            next_id = node_queue.get()
            if (self.nodes[next_id].value == value):
                return next_id  # Success
            for uid in self.neighbors[next_id]:
                if (self.uid_to_index[uid] not in visited):
                    node_queue.put(self.uid_to_index[uid])
                    visited += [self.uid_to_index[uid]]
        return None

    def is_adjacent(self, node_id1, node_id2):
        """Test for adjacency between node1 and node2

         Args:
            node_id1(int): Index of first node
            node_id2(int): Index of second node
        """
        if node_id1 >= self.node_count() or node_id2 >= self.node_count():
            raise IndexError("Node ID out of range for graph")

        return (self.node_uids[node_id1] in self.neighbors[node_id2])

    def get_neighbors(self, node_id):
        """Return list of neighbors of the specified node

        Args:
            node_id: Index of node to examine the neighbors of

        Returns(list): List of current node IDs that are neighbors of node_id.
        """
        if node_id >= self.node_count():
            raise IndexError("Node ID out of range for graph")
        return [self.uid_to_index[i] for i in self.neighbors[node_id]]

    def shortest_path(self, node_id1, node_id2):
        """Find the shortest path between node1 and node2 on the graph

        Args:
            node_id1(int): Index of first node
            node_id2(int): Index of second node

        Returns(list): List of nodes from node_id1 to node_id2 that constitute
        the shortest possible path in the graph between those two nodes.
        """
        if node_id1 >= self.node_count() or node_id2 >= self.node_count():
            raise IndexError("Node ID out of range for graph")

        # Treat special case of equal inputs
        if node_id1 == node_id2:
            return [node_id1]

        # Initialize two arrays for backtracking the shortest path found
        previous = [None] * self.node_count()  # Tracks the moves
        distances = [None] * self.node_count()  # Records distance to nodes
        distances[node_id1] = 0

        node_queue = queue.LifoQueue()
        node_queue.put(node_id1)

        while not node_queue.empty():
            next_id = node_queue.get()
            if next_id == node_id2:
                break  # Success
            new_distance = distances[next_id] + 1
            # On each iteration, check if going to a neighbor node was
            # shortest path to that node, if so, add that node to the queue
            # and record distance and path that is being taken.
            for uid in self.neighbors[next_id]:
                if ((distances[self.uid_to_index[uid]] is None) or
                        (distances[self.uid_to_index[uid]] > new_distance)):
                    distances[self.uid_to_index[uid]] = new_distance
                    previous[self.uid_to_index[uid]] = next_id
                    node_queue.put(self.uid_to_index[uid])

        # Backtrack to get path
        if previous[node_id2] is None:
            raise IndexError("Unable to find target node, "
                             "possibly disconnected graph.")

        path = []
        next = node_id2
        while next is not None:
            path.append(next)
            next = previous[next]
        # Return reversed backtrack path to get in order from node 1 to 2
        return path[::-1]
