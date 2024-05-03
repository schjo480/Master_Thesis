import networkx as nx
import numpy as np
from tqdm import tqdm
import math

'''def calculate_shortest_paths(paths, node_coordinates, edges, distance_measure='edge'):
    G = nx.Graph()
    edge_coordinates = node_coordinates[edges]
    for node, coordinates in enumerate(node_coordinates):
        G.add_node(node, pos=coordinates)
    for edge, coordinates in zip(edges, edge_coordinates):
        euclidean_distance = math.dist(coordinates[0], coordinates[1])
        G.add_edge(*edge, euclidean_distance=euclidean_distance)

    shortest_paths = []
    for path in paths:
        start_edge_idx = path['edge_idxs'][0]
        end_edge_idx = path['edge_idxs'][-1]
        start_edge = edges[start_edge_idx]
        end_edge = edges[end_edge_idx]
        start_node = start_edge[0]
        end_node = end_edge[1]
        if distance_measure == 'edge':
            shortest_path = nx.shortest_path(G, start_node, end_node)
        elif distance_measure == 'euclidean':
            shortest_path = nx.shortest_path(G, start_node, end_node, weight='euclidean_distance')
        shortest_paths.append(shortest_path)
    return shortest_paths'''

def calculate_shortest_paths(paths, node_coordinates, edges, distance_measure='edge', multiple_paths=False):
    G = nx.Graph()
    edge_coordinates = node_coordinates[edges]
    for node, coordinates in enumerate(node_coordinates):
        G.add_node(node, pos=coordinates)
    for edge, coordinates in zip(edges, edge_coordinates):
        euclidean_distance = math.dist(coordinates[0], coordinates[1])
        G.add_edge(*edge, euclidean_distance=euclidean_distance)

    shortest_paths = []

    for path in tqdm(paths):
        shortest_path = {}
        for key in path.keys():
            shortest_path[key] = path[key]
            
        start_edge_idx = path['edge_idxs'][0]
        end_edge_idx = path['edge_idxs'][-1]
        start_edge = edges[start_edge_idx]
        end_edge = edges[end_edge_idx]
        start_node = start_edge[0] if path['edge_orientations'][0] == 1 else start_edge[1]
        end_node = end_edge[1] if path['edge_orientations'][-1] == 1 else end_edge[0]
        if distance_measure == 'edge':
            if multiple_paths:
                shortest_path_nodes = nx.all_shortest_paths(G, start_node, end_node)
                shortest_path_edge_idxs = []
                shortest_path_edge_orientation = [] 
                
                for shortest_path_int in shortest_path_nodes:
                    shortest_path_edge_idxs_int = []
                    shortest_path_edge_orientation_int = [] 
                    for i in range(len(shortest_path_int) - 1):
                        node1 = shortest_path_int[i]
                        node2 = shortest_path_int[i + 1]
                        edge_idx = np.where(((edges[:, 0] == node1) & (edges[:, 1] == node2)) | (edges[:, 0] == node2) & (edges[:, 1] == node1))[0]
                        shortest_path_edge_idxs_int.append(edge_idx[0])
                        shortest_path_edge_orientation_int.append(1 if edges[edge_idx][0][0] == node1 else -1)
                    shortest_path_edge_idxs.append(shortest_path_edge_idxs_int)
                    shortest_path_edge_orientation.append(shortest_path_edge_orientation_int)
                
            else:
                shortest_path_nodes = nx.shortest_path(G, start_node, end_node)
                shortest_path_edge_idxs = []
                shortest_path_edge_orientation = [] 

                for i in range(len(shortest_path_nodes) - 1):
                    node1 = shortest_path_nodes[i]
                    node2 = shortest_path_nodes[i + 1]
                    edge_idx = np.where(((edges[:, 0] == node1) & (edges[:, 1] == node2)) | (edges[:, 0] == node2) & (edges[:, 1] == node1))[0]
                    shortest_path_edge_idxs.append(edge_idx[0])
                    shortest_path_edge_orientation.append(1 if edges[edge_idx][0][0] == node1 else -1)

        elif distance_measure == 'euclidean':
            if multiple_paths:
                shortest_path_nodes = nx.all_shortest_paths(G, start_node, end_node, weight='euclidean_distance')
                shortest_path_edge_idxs = []
                shortest_path_edge_orientation = [] 
                
                for shortest_path_int in shortest_path_nodes:
                    shortest_path_edge_idxs_int = []
                    shortest_path_edge_orientation_int = [] 
                    for i in range(len(shortest_path_int) - 1):
                        node1 = shortest_path_int[i]
                        node2 = shortest_path_int[i + 1]
                        edge_idx = np.where(((edges[:, 0] == node1) & (edges[:, 1] == node2)) | (edges[:, 0] == node2) & (edges[:, 1] == node1))[0]
                        shortest_path_edge_idxs_int.append(edge_idx[0])
                        shortest_path_edge_orientation_int.append(1 if edges[edge_idx][0][0] == node1 else -1)
                    shortest_path_edge_idxs.append(shortest_path_edge_idxs_int)
                    shortest_path_edge_orientation.append(shortest_path_edge_orientation_int)
                    
            else:
                shortest_path_nodes = nx.shortest_path(G, start_node, end_node, weight='euclidean_distance')
                shortest_path_edge_idxs = []
                shortest_path_edge_orientation = []  

                for i in range(len(shortest_path_nodes) - 1):
                    node1 = shortest_path_nodes[i]
                    node2 = shortest_path_nodes[i + 1]
                    edge_idx = np.where(((edges[:, 0] == node1) & (edges[:, 1] == node2)) | ((edges[:, 0] == node2) & (edges[:, 1] == node1)))[0]
                    shortest_path_edge_idxs.append(edge_idx[0])  # Modify here to append the first element of the array
                    shortest_path_edge_orientation.append(1 if edges[edge_idx][0][0] == node1 else -1)
        shortest_path['edge_idxs'] = shortest_path_edge_idxs
        shortest_path['edge_orientation'] = shortest_path_edge_orientation 

        shortest_paths.append(shortest_path)

    return shortest_paths


# nodes of last shortest path: [4577, 4548, 5406, 1664, 1780, 8511, 1952, 1489, 567, 4289, 1519, 1929, 8444, 9632, 1348, 4895, 7929, 3407, 4014, 5290, 4556, 4317, 4197, 4219, 4721]
# edges of last actual path: 
'''[4577 8261]
 [4577 9015]
 [5412 9015]
 [4103 5412]
 [1664 4103]
 [1664 1780]
 [1780 8511]
 [1952 8511]
 [1952 4368]
 [4367 4368]
 [4367 4369]
 [2649 4369]
 [2649 2681]
 [2681 6838]
 [2003 6838]
 [2003 4468]
 [4468 4724]
 [4718 4724]
 [4718 9126]
 [4588 9126]
 [4571 4588]
 [4464 4571]
 [4464 4895]
 [4895 9125]
 [4896 9125]
 [4896 9171]
 [4716 9171]
 [3533 4716]
 [3533 4708]
 [1838 4708]
 [1838 4641]
 [1894 4641]
 [1894 4477]
 [4359 4477]
 [4359 4360]
 [4360 9333]
 [9131 9333]
 [4726 9131]
 [4721 4726]
 [4219 4721]'''

# calculate the differences in length between actual and shortest paths