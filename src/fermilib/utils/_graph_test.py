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

import unittest

from fermilib.utils import Graph, Node


class GraphTest(unittest.TestCase):

    def test_two_node_graph(self):
        """Build a graph with two nodes and one edge"""
        two_node = Graph()
        two_node.add_node()
        two_node.add_node()
        two_node.add_edge(0, 1)

        # Check to see if it built normally
        self.assertEqual(two_node.neighbors,
                         [{1}, {0}])
        self.assertEqual(two_node.node_uids,
                         [0, 1])

        self.assertEqual(two_node.uid_to_index[0],
                         0)
        self.assertEqual(two_node.uid_to_index[1],
                         1)
        self.assertEqual(two_node.node_count(), 2)

        # Now remove an edge, then a node, see if it
        two_node.remove_edge(0, 1)
        self.assertEqual(two_node.neighbors,
                         [set([]), set([])])
        two_node.remove_node(0)
        self.assertEqual(two_node.node_uids, [1])

        # Add a node and edge back, verify indexing
        two_node.add_node()
        two_node.add_edge(0, 1)
        self.assertEqual(two_node.neighbors,
                         [{2}, {1}])
        self.assertEqual(two_node.uid_to_index[1], 0)
        self.assertEqual(two_node.uid_to_index[2], 1)
        self.assertEqual(two_node.node_count(), 2)

        self.assertEqual(two_node.get_neighbors(0), [1])
        self.assertEqual(two_node.get_neighbors(1), [0])

        self.assertTrue(two_node.is_adjacent(0, 1))

        # Check path between nodes
        self.assertEqual(two_node.shortest_path(0, 1), [0, 1])
        self.assertEqual(two_node.shortest_path(1, 0), [1, 0])
        self.assertEqual(two_node.shortest_path(0, 0), [0])

        # Remove just the node and check neighbors
        two_node.remove_node(1)
        self.assertEqual(two_node.neighbors,
                         [set([])])

    def test_ring(self):
        """Build a ring of 8 nodes and test path finding"""
        eight_node = Graph()
        for i in range(8):
            eight_node.add_node(Node(value=i))
        for i in range(8):
            eight_node.add_edge(i, (i + 1) % 8)
        self.assertEqual(eight_node.neighbors,
                         [{1, 7}, {0, 2}, {1, 3}, {2, 4},
                          {3, 5}, {4, 6}, {5, 7}, {0, 6}])

        self.assertEqual(eight_node.shortest_path(0, 4),
                         [0, 7, 6, 5, 4])
        self.assertEqual(eight_node.shortest_path(0, 6),
                         [0, 7, 6])
        self.assertTrue(eight_node.is_adjacent(0, 7))

        # Look for node with value 6 and value not present
        found_index = eight_node.find_index(6)
        self.assertEqual(found_index, 6)
        found_index = eight_node.find_index(10)
        self.assertIsNone(found_index)

        # Make a hole in the ring and check new distance
        eight_node.remove_node(7)
        self.assertTrue(eight_node.is_adjacent(0, 1))
        self.assertFalse(eight_node.is_adjacent(0, 6))
        self.assertEqual(eight_node.shortest_path(0, 6),
                         [0, 1, 2, 3, 4, 5, 6])

# Run test.
if __name__ == '__main__':
    unittest.main()
