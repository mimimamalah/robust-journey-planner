from pyspark.sql import SparkSession
from pyspark.sql.functions import col 
import networkx as nx
import pandas as pd
from itertools import islice
import itertools
from time_utils import *
from probability_computing import *

MAX_NUMBER_OF_PATHS = 100

# +
def calculate_paths_info(graph, start_node, end_node, max_paths=MAX_NUMBER_OF_PATHS, weight="expected_travel_time"):
    
    # Calculate all shortest paths from start_node to end_node
    shortest_paths_generator = nx.all_shortest_paths(graph, start_node, end_node, weight=weight)
    
    # Retrieve up to max_paths shortest paths
    shortest_paths = list(itertools.islice(shortest_paths_generator, max_paths))
  
    paths_info_list = []
    
    # For each shortest path
    for path in shortest_paths:
        path_info = []
        # Iterate through the nodes in the path, excluding the start and end nodes
        for node in path[1:-1]:
            # Collect information for each edge
            start_time = graph.nodes[node]["start_time"]
            end_time = graph.nodes[node]["end_time"]
            start_stop_id = graph.nodes[node]["start_stop_id"]
            end_stop_id = graph.nodes[node]["end_stop_id"]
            trip_id = graph.nodes[node]["trip_id"]
            
            edge_info = {
                "start_time": get_time(start_time),
                "end_time": get_time(end_time),
                "start_stop_id": start_stop_id,
                "end_stop_id": end_stop_id,
                "trip_id": trip_id
            }
            path_info.append(edge_info)
        
        paths_info_list.append(path_info)

    return paths_info_list

def add_node_to_graph(graph, index, att):
    # Add the node to the graph with the given attributes
    graph.add_node(
        index, 
        start_stop_id=att['start_stop_id'], 
        end_stop_id=att['end_stop_id'], 
        start_time=att['start_time'], 
        end_time=att['end_time'], 
        expected_travel_time=att['expected_travel_time'],
        trip_id=att['trip_id']
    )

def add_walking_nodes_to_graph(graph, start_node, end_node, node_sequence, edge_df, node_attrs, node_index):
    
    def create_node_attributes(start_time, end_time, node_attrs):
        return {
            "start_time": start_time,
            "end_time": end_time,
            "expected_travel_time": node_attrs["expected_travel_time"],
            "start_stop_id": node_attrs["start_stop_id"],
            "end_stop_id": node_attrs["end_stop_id"],
            "trip_id": "None"
        }

    if start_node == node_sequence[0][0]:
        # Handling the case when the start node is the same as the first node in node_sequence
        subsequent_connections = edge_df[(edge_df["start_stop_id"] == end_node) & (edge_df["is_walking"] == 0)]
        
        for sub_node, edge in subsequent_connections.iterrows():
            walk_duration = node_attrs["expected_travel_time"]
            walk_end_time = edge["start_time"]
            walk_start_time = walk_end_time - walk_duration
            
            node_data = create_node_attributes(walk_start_time, walk_end_time, node_attrs)
            graph.add_node(sub_node+node_index, **node_data)
            node_index += 1
            
    else:
        # Handling the case for previous connections where the end_stop_id matches the start_stop_id of the current node attributes
        previous_connections = [
            (node, attr) for node, attr in graph.nodes(data=True) 
            if attr["end_stop_id"] == node_attrs["start_stop_id"]
        ]
        
        for prev_node, prev_attrs in previous_connections:
            walk_duration = node_attrs["expected_travel_time"]
            walk_start_time = prev_attrs["end_time"]
            walk_end_time = walk_start_time + walk_duration
            
            node_data = create_node_attributes(walk_start_time, walk_end_time, node_attrs)
            graph.add_node(prev_node+node_index, **node_data)
            node_index += 1
            
    return node_index

def get_path(extracted_nodes, edges):
    """Given a dataframe of edges and pairs of nodes, return feasible paths with the same node tuple sequence as in extracted_nodes.

    Args:
        extracted_nodes (list): Ordered pairs of nodes representing the desired edges.
        edges (DataFrame): All available edges in the graph.

    Returns:
        paths (list): All feasible paths with the same node tuple sequence as in extracted_nodes.
    """
    
    # Initialize a directed graph
    graph= nx.DiGraph()

    # Add connections to graph
    add_connections_to_graph(graph, extracted_nodes, edges)
    
    if not add_start_and_end_nodes(graph, extracted_nodes):
        return None


    add_edges_between_nodes(graph)
    
    # Check if a path exists and return the path info
    if nx.has_path(graph, "start", "end"):
        return calculate_paths_info(graph, "start", "end")
    else:
        return None

def add_connections_to_graph(graph, extracted_nodes, edges):
    """Add nodes and edges for given connections to the graph."""
    for start, end in extracted_nodes:
        connections = edges[edges["start_stop_id"] == start]
        i = 100000000

        for index, att in connections.iterrows():
            if att['is_walking'] == 0:
                add_node_to_graph(graph, index, att)
            else:
                i = add_walking_nodes_to_graph(graph, start, end, extracted_nodes, edges, att, i)

def add_start_and_end_nodes(graph, node_sequence):
    """Add start and end nodes to the graph with edges connecting to actual start and end nodes in the path.

    Args:
        graph (networkx.DiGraph): The graph to which the nodes will be added.
        node_sequence (list): The sequence of nodes extracted.

    Returns:
        bool: True if nodes and edges were added successfully, False otherwise.
    """
    
    start_node_id = node_sequence[0][0]
    end_node_id = node_sequence[-1][1]

    # Find nodes in the graph that match the start node ID
    start_node_connections = [
        (node, attr) for node, attr in graph.nodes(data=True) 
        if attr["start_stop_id"] == start_node_id
    ]

    if not start_node_connections:
        return False

    # Add edges from the "start" node to the matched nodes
    for node, attr in start_node_connections:
        graph.add_edge("start", node, expected_travel_time=0)

    # Find nodes in the graph that match the end node ID
    end_node_connections = [
        (node, attr) for node, attr in graph.nodes(data=True) 
        if node != "start" and attr["end_stop_id"] == end_node_id
    ]

    if not end_node_connections:
        return False

    # Add edges from the matched nodes to the "end" node
    for node, attr in end_node_connections:
        graph.add_edge(node, "end", expected_travel_time=attr["expected_travel_time"])

    return True

# +

def add_edges_between_nodes(graph):
    """Add edges between nodes in the graph based on the specified conditions."""
    for source_node_id, source_node_attrs in graph.nodes(data=True):
        for target_node_id, target_node_attrs in graph.nodes(data=True):
            if source_node_id == target_node_id:
                continue
            
            if source_node_attrs and target_node_attrs:
                if source_node_attrs['end_stop_id'] == target_node_attrs['start_stop_id']: 
                    # Condition for the same trip
                    if ((source_node_attrs['end_time'] <= target_node_attrs['start_time']) or 
                        (abs(source_node_attrs['end_time'] - target_node_attrs['start_time']) < 1)) and \
                        source_node_attrs['trip_id'] == target_node_attrs['trip_id']: 
                        graph.add_edge(source_node_id, target_node_id, expected_travel_time=(target_node_attrs['start_time'] - source_node_attrs['start_time']))
                    # Condition for different trips
                    elif ((source_node_attrs['end_time'] < target_node_attrs['start_time']) and 
                          (abs(source_node_attrs['end_time'] - target_node_attrs['start_time']) >= 120)) and \
                          source_node_attrs['trip_id'] != target_node_attrs['trip_id']:
                        graph.add_edge(source_node_id, target_node_id, expected_travel_time=(target_node_attrs['start_time'] - source_node_attrs['start_time']))


def get_k_fastest_path(paths, k):
    """Select the k fastest paths in a list

    Args:
        paths: list of list of dictionnary representing edges
        k: number of paths to keep

    Returns:
        shortest_paths: The k shortest paths
    """
    #Compute the time for each path
    '''
    valid_paths = []
    for path in paths:
        valid = True
        for i in range(len(path) - 1):
            if i + 1 < len(path):
                current_edge = path[i]
                next_edge = path[i + 1]
                if next_edge['start_time'] == current_edge['end_time'] and next_edge['trip_id'] != 'None':
                    valid = False
        if valid :
            valid_paths.append(path)
    '''
    paths_with_durations = []
    
    # Compute the total travel time for each path
    for path in paths:
        total_duration = calculate_total_time(path)
        paths_with_durations.append((total_duration, path))
    paths_with_durations.sort(key=lambda x: x[0])
    
    # Select best paths
    fastest_paths_with_durations = paths_with_durations[:k]
    
    fastest_paths = [path for total_duration, path in fastest_paths_with_durations]
    
    return fastest_paths


def paths_confidence(graph, paths_nodes, df_edges, confidence_interval, df_avg_delay):
    """
    Only keep the paths having confidence above confidence_interval,
    Here we used considered that we should be confident about the path between
    the source and sink excluded.
    """
    confidence_paths = []
    
    # Compute the confidence for each path
    for path in paths_nodes:
        
        # Remove the edges from source and to sink
        path_s = path[1:-1]

        # Compute the confidence of the path
        probability_exp, probability_norm = calculate_connection_probability(path_s, df_avg_delay)

        # We used exponential probability since it yielded better results
        if probability_exp >= confidence_interval:
            confidence_paths.append(path)
            
    return confidence_paths


def get_best_paths(df_edges, source, sink, desired_arrival_time, max_trip_length, k, confidence_interval, df_avg_delay):
    
    desired_arrival_seconds = get_sec(desired_arrival_time)
    
    filtered_edges = df_edges[(df_edges['is_walking'] == 1) | ((df_edges['end_time'] <= desired_arrival_seconds) & (df_edges['end_time'] > desired_arrival_seconds - max_trip_length))]   
    graph = nx.from_pandas_edgelist(filtered_edges, 'start_stop_id', 'end_stop_id', edge_attr=['expected_travel_time'], create_using=nx.DiGraph)
    fastest_paths_generator = nx.shortest_simple_paths(graph, source, sink, weight='expected_travel_time')
    fastest_paths = list(islice(fastest_paths_generator, 100))


    collected_paths = []
    for path in fastest_paths:
        node_pairs = list(zip(path[:-1], path[1:]))
        node_pairs_df = pd.DataFrame(node_pairs, columns=["start_stop_id", "end_stop_id"])
        filtered_edges_for_path = filtered_edges.merge(node_pairs_df, on=["start_stop_id", "end_stop_id"], how="inner")
        
        # Get the path if it exists.
        path_segments = get_path(node_pairs, filtered_edges_for_path)
        if path_segments:
            for i, segment in enumerate(path_segments):
                if i < MAX_NUMBER_OF_PATHS:
                    collected_paths.append(segment)
                    
    # Remove None values from the collected paths
    valid_paths = [path for path in collected_paths if path is not None]
    
    # Remove duplicate paths
    unique_paths = remove_duplicate_paths(valid_paths)
    
    # Compress trips in each unique path.
    compressed_paths = [compress_trips(trip) for trip in unique_paths]

    confidence_paths = paths_confidence(graph, compressed_paths, df_edges, confidence_interval, df_avg_delay)
    
    # Get the k fastest paths
    fastest_k_paths = get_k_fastest_path(confidence_paths, k)
    print(fastest_k_paths)
    
    return fastest_k_paths

def remove_duplicate_paths(paths):
    path_dict_by_time = {}
    for path in paths:
        initial_time = path[0]['start_time']
        final_time = path[-1]['end_time']
        time_key = (initial_time, final_time)
        path_dict_by_time[time_key] = path

    # Extract the unique paths from the dictionary
    unique_paths = list(path_dict_by_time.values())
    return unique_paths

def compress_trips(trip_edges):
    compressed_edges = []
    
    for current_edge in trip_edges:
        # Append non-compressible edges or edges with different trip_id or None trip_id
        if not compressed_edges or current_edge['trip_id'] != compressed_edges[-1]['trip_id'] or (current_edge['trip_id'] is None and compressed_edges[-1]['trip_id'] is None):
            compressed_edges.append(current_edge.copy())
        else:
            # Compress edges with the same trip_id
            compressed_edges[-1]['end_time'] = current_edge['end_time']
            compressed_edges[-1]['end_stop_id'] = current_edge['end_stop_id']
    
    return compressed_edges
