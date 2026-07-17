import community.community_louvain as community_louvain
import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
import time

def outer_edge_strategy(g: nx.Graph, pos_edges: list[tuple[int, int, any]],percentuale_negativi = 1.0) -> set[tuple[int, int]]:
    """
    Selects edges where both the source and destination nodes are different from those
    involved in the given positive relations.

    Parameters:
        g (nx.Graph):
            The graph containing the nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            A list of positive edges, where each edge is represented as a tuple
            (source, destination, label). The label is ignored in this function.

    Returns:
        set (set[tuple[int, int]]):
            A set of predicted negative edges, including all possible edges between nodes
            that are neither sources nor destinations in `pos_edges`.

    Description:
        - Extracts the sets of source nodes (`pos_sources`) and destination nodes (`pos_dests`)
        from `pos_edges`.
        - Identifies all nodes present in the graph.
        - Computes the Cartesian product of nodes, excluding those that belong to
        `pos_sources` or `pos_dests`.
        - Returns this set as predicted negative edges.
    """

    max_negatives = len(pos_edges) * 3
    pos_sources = {s for s, _, _ in pos_edges}
    pos_dests = {d for _, d, _ in pos_edges}
    nodes = set(g.nodes())

    predicted_negatives = set((source, dest) for source in nodes - pos_sources for dest in nodes - pos_dests)

    #return predicted_negatives
    return random.sample(list(predicted_negatives), max_negatives) if  len(predicted_negatives) > max_negatives else predicted_negatives

def common_node_strategy(g: nx.Graph, pos_edges: list[tuple[int, int, any]],percentuale_negativi = 1.0) -> set[tuple[int, int]]:
    """
    Selects edges where exactly one endpoint (source or destination) is involved in the given positive relations.

    Parameters:
        g (nx.Graph):
            The graph containing the nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            A list of positive edges, where each edge is represented as a tuple
            (source, destination, label). The label is ignored in this function.

    Returns:
        set (set[tuple[int, int]]):
            A set of predicted negative edges, including all possible edges where one
            node is in `pos_sources` but not in `pos_dests`, or vice versa.

    Description:
        - Extracts the sets of source nodes (`pos_sources`) and destination nodes (`pos_dests`)
          from `pos_edges`.
        - Identifies all nodes present in the graph.
        - Constructs edges where one node belongs to `pos_sources` but not `pos_dests`, and vice versa.
        - Returns this set as predicted negative edges.
    """
    max_negatives = len(pos_edges) * 3
    pos_sources = {s for s, _, _ in pos_edges}
    pos_dests = {d for _, d, _ in pos_edges}
    nodes = set(g.nodes())

    predicted_negatives = {(source, dest) for source in pos_sources for dest in nodes - pos_dests} | \
                          {(source, dest) for source in nodes - pos_sources for dest in pos_dests}

    return random.sample(list(predicted_negatives), max_negatives) if  len(predicted_negatives) > max_negatives else predicted_negatives
    #return set(random.sample(list(predicted_negatives), max_negatives)) if ((len(predicted_negatives) > max_negatives) and (max_negatives != -1)) else predicted_negatives

import random
from tqdm import tqdm
import networkx as nx
from typing import Any

def sampled_common_node_strategy(
    g: nx.Graph,
    pos_edges: list[tuple[int, int, Any]],
    percentuale_negativi: float,
    number_of_negatives:int
) -> set[tuple[int, int]]:
    """
    Selects negative edges by sampling from node pairs where exactly one node is involved
    in positive edges, limited to a total number of max_negatives.
    """
    random.seed(42)

    max_negatives = int(len(pos_edges) * percentuale_negativi)
    pos_sources = {s for s, _, _ in pos_edges}
    pos_dests = {d for _, d, _ in pos_edges}
    nodes = set(g.nodes())

    predicted_negatives = set()
    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    # Candidate edges where source ∈ pos_sources, dest ∉ pos_dests
    for source in pos_sources:
        for dest in nodes - pos_dests:
            if len(predicted_negatives) >= max_negatives:
                break
            edge = (source, dest)
            if not g.has_edge(*edge):
                predicted_negatives.add(edge)
                pbar.update(1)
        if len(predicted_negatives) >= max_negatives:
            break

    # Candidate edges where source ∉ pos_sources, dest ∈ pos_dests
    for source in nodes - pos_sources:
        for dest in pos_dests:
            if len(predicted_negatives) >= max_negatives:
                break
            edge = (source, dest)
            if not g.has_edge(*edge):
                predicted_negatives.add(edge)
                pbar.update(1)
        if len(predicted_negatives) >= max_negatives:
            break

    pbar.close()
    #return predicted_negatives
    return random.sample(predicted_negatives, number_of_negatives) if  len(predicted_negatives) > number_of_negatives else predicted_negatives


def sampled_common_node_strategy_limited(
    g: nx.Graph,
    pos_edges: list[tuple[int, int, Any]],
    percentuale_negativi: float,
    number_of_negatives: int
) -> set[tuple[int, int]]:
    """
    Selects negative edges by sampling from node pairs where exactly one node is involved
    in positive edges, limited to a total number of max_negatives.
    """
    random.seed(42)

    max_negatives = int(len(pos_edges)*3)
    pos_sources = {s for s, _, _ in pos_edges}
    pos_dests = {d for _, d, _ in pos_edges}
    nodes = list(g.nodes())

    predicted_negatives = set()
    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    # Combine both types of candidate sources
    candidate_sources = list(pos_sources) + list(set(nodes) - pos_sources)
    random.shuffle(candidate_sources)

    outer_iterations = 0
    for source in candidate_sources:
        if len(predicted_negatives) >= max_negatives or outer_iterations >= max_negatives:
            break

        # Decide if source is from pos_sources or not
        if source in pos_sources:
            candidates = list(set(nodes) - pos_dests)
        else:
            candidates = list(pos_dests)

        random.shuffle(candidates)
        for dest in candidates:
            if len(predicted_negatives) >= max_negatives:
                break
            if source != dest and not g.has_edge(source, dest):
                predicted_negatives.add((source, dest))
                pbar.update(1)
        outer_iterations += 1

    pbar.close()
    return predicted_negatives
    #return random.sample(predicted_negatives, number_of_negatives) if  len(predicted_negatives) > number_of_negatives else predicted_negatives



def topological_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int, int, any]],percentuale_negativi = 1.0) -> set[tuple[int, int]]:
    """
    Generates negative edges by selecting node pairs that are topologically close
    to positive edges but do not have an existing edge.

    Parameters:
        g (nx.Graph):
            The graph containing the nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            A list of positive edges, where each edge is represented as a tuple
            (source, destination, label). The label is ignored in this function.

    Returns:
        set (set[tuple[int, int]]):
            A set of predicted negative edges.

    Description:
        - Identifies nodes that are topologically close to those involved in `pos_edges`.
        - Selects pairs of nodes that are near positive edges but not directly connected.
        - Ensures that the number of generated negative edges does not exceed `max_negatives`.
        - Returns the selected node pairs as predicted negative edges.
    """
    random.seed(42)


    max_negatives = len(pos_edges) * 3
    pos_sources = {s for s, _, _ in pos_edges}
    pos_dests = {d for _, d, _ in pos_edges}



    negative_edges = set()
    i = 0

    sample = pos_sources | pos_dests

    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    while len(negative_edges) <max_negatives:
        # Pick a positive node as reference
        source = random.choice(list(sample))
        # Select a neighbor of the source node
        neighbors = set(g.neighbors(source))
        if not neighbors:  # Skip if the node has no neighbors
            continue

        dest = random.choice(list(neighbors))

        # Make sure it's not a positive edge
        if source != dest:
            negative_edges.add((source, dest))

        i += 1
        pbar.update(1)
        if i == max_negatives:
            break

    #return random.sample(negative_edges, int(len(negative_edges)*percentuale_negativi))
    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges

def random_negative_sampling(
    g: nx.Graph,
    pos_edges: list[tuple[int, int, any]],
    type_src: str,
    type_dst: str,
    percentuale_negativi: float = 1.0
) -> set[tuple[int, int]]:
    """
    Generates random negative edges, without considering node degrees.

    Parameters:
        g (nx.Graph): Input graph.
        pos_edges (list[tuple[int, int, Any]]): Positive edges.
        type_src (str): Label of source node type to consider.
        type_dst (str): Label of destination node type to consider.
        percentuale_negativi (float): Multiplier for #negatives (default=1.0).

    Returns:
        set[tuple[int, int]]: Set of negative edges.
    """

    random.seed(42)
    src_nodes = [n for n, d in g.nodes(data=True) if d["label"] == type_src]
    dst_nodes = [n for n, d in g.nodes(data=True) if d["label"] == type_dst]

    max_negatives = int(len(pos_edges))
    negative_edges = set()
    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    iter_c = 0
    prev_len = 0
    while len(negative_edges) < max_negatives:
        u = random.choice(src_nodes)
        v = random.choice(dst_nodes)

        if u != v and not g.has_edge(u, v):
            negative_edges.add((u, v))
            pbar.update(1)

        if prev_len != len(negative_edges):
            iter_c = 0
            prev_len = len(negative_edges)

        # break if after 10,000,000 iterations if no negative generated
        if iter_c > 10_000_000 and prev_len == len(negative_edges):
            break

        iter_c += 1

    pbar.close()
    return negative_edges

def degree_aware_negative_sampling(
    g: nx.Graph,
    pos_edges: list[tuple[int, int, any]],
    type_src: str,
    type_dst: str,
    num_buckets: int = 10,
    percentuale_negativi: float = 1.0
) -> set[tuple[int, int]]:
    """
    Generates degree-aware negative edges using a progressive bucket strategy.

    Steps:
        1. Divide nodes into degree-based buckets.
        2. Sample negatives within top-degree buckets first.
        3. If not enough negatives, progressively sample between top buckets.
        4. Ensure negatives do not exist in the graph and match node types.

    Parameters:
        g (nx.Graph): Input graph.
        pos_edges (list[tuple[int, int, Any]]): Positive edges.
        type_src (str): Label of source node type to consider.
        type_dst (str): Label of destination node type to consider.
        num_buckets (int): Number of degree buckets (default=5).
        percentuale_negativi (float): Multiplier for #negatives (default=1.0).

    Returns:
        set[tuple[int, int]]: Set of negative edges.
    """

    random.seed(42)
    degrees = dict(g.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

    # Create degree buckets
    buckets = {i: [] for i in range(num_buckets)}
    for i, node in enumerate(sorted_nodes):
        bucket_idx = int(i / (len(sorted_nodes) / num_buckets))
        buckets[bucket_idx].append(node)

    node_to_bucket = {node: b for b, nodes in buckets.items() for node in nodes}

    max_negatives = int(3 * len(pos_edges))
    negative_edges = set()
    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    def try_add_negative(u, v_candidate):
        """Helper to add a valid negative edge if conditions hold."""
        if (
            u != v_candidate
            and not g.has_edge(u, v_candidate)
            and g.nodes[u]["label"] == type_src
            and g.nodes[v_candidate]["label"] == type_dst
        ):
            negative_edges.add((u, v_candidate))
            pbar.update(1)

    # Define progressive bucket pairings
    top_buckets = list(buckets.keys())[:num_buckets]
    bucket_pairs = [(i, j) for i in range(num_buckets) for j in range(i, num_buckets)]

    max_attempts = 10 * max_negatives  # safety limit
    attempts = 0
    prev_size = 0

    while len(negative_edges) < max_negatives and attempts < max_attempts:
        for b1, b2 in bucket_pairs:
            current_batch = set()

            candidates_u = buckets[b1]
            candidates_v = buckets[b2]
            if not candidates_u or not candidates_v:
                continue

            u = random.choice(candidates_u)
            v_prime = random.choice(candidates_v)
            try_add_negative(u, v_prime)
            current_batch.add((u, v_prime))

            if len(negative_edges) > max_negatives:
                excess = len(negative_edges) - max_negatives
                to_remove = random.sample(list(current_batch), min(excess, len(current_batch)))
                negative_edges.difference_update(to_remove)
                break

        # progress tracking
        if len(negative_edges) == prev_size:
            attempts += 1
        else:
            attempts = 0
        prev_size = len(negative_edges)

    pbar.close()
    if attempts >= max_attempts:
        print(f"[WARNING] Negative sampling stopped early at {len(negative_edges)} samples.")

    return negative_edges


def degree_threshold_based_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int,int,any]], percentuale_negativi = 1.0, degree_threshold: int =2) -> set[tuple[int, int]]:

    """
    Generates negative edges by sampling node pairs with controlled degree differences,
    excluding existing connections and low-degree nodes.

    Parameters:
        g (nx.Graph):
            The graph containing nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            List of positive edges as tuples (source, destination, label).
            Label is ignored.

        degree_threshold (int):
            Minimum degree requirement for node eligibility (default=2).

    Returns:
        set (set[tuple[int, int]]):
            Set of predicted negative edges.

    Description:
        - Filters nodes based on degree threshold (>= specified value)
        - Samples node pairs from eligible nodes while:
            1. Avoiding existing connections
            2. Maintaining degree difference below 10 units
        - Prioritizes nodes with moderate degree similarities
        - Returns when reaching target count or after max iterations
    """
    random.seed(42)

    # Extract node degrees
    node_degrees = dict(g.degree())

    # Filter out nodes with low degree
    valid_nodes = [node for node, degree in node_degrees.items() if degree >= degree_threshold]

    negative_edges = set()


    max_negatives = len(pos_edges)*3
    pbar = tqdm(total=max_negatives, dynamic_ncols=True)
    i = 0

    while len(negative_edges) < max_negatives:
        u, v = random.sample(valid_nodes, 2)  # Take two random nodes with a valid degree

        if u != v:  # Make sure there isn't already an edge
            # Check whether the degrees are too far apart
            if abs(node_degrees[u] - node_degrees[v]) < 10:  # Set a threshold for the degree difference
                negative_edges.add((u, v))

        i += 1
        pbar.update(1)
        if i == max_negatives: # Let's reconsider this
            break

    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges





def edge_betweenness_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int,int,any]],percentuale_negativi = 1.0) -> set[tuple[int, int]]:

    """
    Generates negative edges by sampling node pairs connected to edges with high edge betweenness centrality,
    excluding existing edges.

    Parameters:
        g (nx.Graph):
            The graph containing nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            List of positive edges as tuples (source, destination, label).
            Label is ignored.


    Returns:
        set (set[tuple[int, int]]):
            Set of predicted negative edges.

    Description:
        - Computes edge betweenness centrality for all edges in the graph
        - Selects nodes connected by the top edges with highest betweenness centrality
        - Samples node pairs from these high-betweenness nodes ensuring:
            1. No existing edge between the pair
            2. Nodes are distinct
        - Returns when reaching target count or after max iterations
    """
    random.seed(42)

    # Compute the edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(g)

    max_negatives = len(pos_edges)*3



    # Sort edges by descending betweenness
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

    # Extract nodes connected by high-betweenness edges
    high_betweenness_nodes = set()
    for x, _ in sorted_edges[:max_negatives]:  # Take the top most central edges
        u,v,_ = x
        high_betweenness_nodes.add(u)
        high_betweenness_nodes.add(v)

    # Convert to list for fast sampling
    high_betweenness_nodes = list(high_betweenness_nodes)

    negative_edges = set()
    pbar = tqdm(total=max_negatives, dynamic_ncols=True)
    i = 0


    while len(negative_edges) < max_negatives:
        u, v = random.sample(high_betweenness_nodes, 2)  # Take two random nodes among those selected

        if u != v:
            negative_edges.add((u, v))

        i += 1
        pbar.update(1)
        if i == max_negatives: # Let's reconsider this
            break

    #return random.sample(negative_edges, int(len(negative_edges)*percentuale_negativi))
    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges


def community_based_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int,int,any]],type_src,type_dst,percentuale_negativi = 1.0) -> set[tuple[int, int]]:

    """
    Generates negative edges by sampling node pairs from different communities detected in the graph,
    ensuring no existing edges between them.

    Parameters:
        g (nx.Graph):
            The graph containing nodes and edges. If directed, it is converted to undirected for community detection.

        pos_edges (list[tuple[int, int, Any]]):
            List of positive edges as tuples (source, destination, label).
            Label is ignored.

    Returns:
        set (set[tuple[int, int]]):
            Set of predicted negative edges.

    Description:
        - Converts directed graphs to undirected for compatibility with Louvain community detection
        - Detects communities using the Louvain method
        - Groups nodes by their community membership
        - Samples negative edges by selecting node pairs from different communities
        - Ensures sampled edges do not exist in the positive edge set and nodes are distinct
        - Returns when reaching target count or after max iterations
    """
    random.seed(42)

    # If the graph is directed, convert it to undirected since louvain doesn't support directed graphs
    if g.is_directed():
        g_und = g.to_undirected()
    else:
        g_und = g

    # Detect communities in the graph
    partition = community_louvain.best_partition(g_und)

    # Groups of nodes by community
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    # Select edges between nodes from different communities
    negative_edges = set()

    print(f'Number of communities: {len(communities)}')

    max_negatives = len(pos_edges)

    pbar = tqdm(desc = "Predicting negatives: ", total=max_negatives, dynamic_ncols=True)
    i = 0

    while len(negative_edges) < max_negatives:
        community_ids = list(communities.keys())
        # print(community_ids)
        # break
        if len(community_ids) < 2:
            break

        comm1, comm2 = random.sample(community_ids, 2)
        u = random.choice(communities[comm1])
        v = random.choice(communities[comm2])

        if u != v and g.nodes[u]["label"] == type_src and g.nodes[v]["label"] == type_dst and not g.has_edge(u, v):
            negative_edges.add((u, v))
            i += 1


        pbar.update(1)
        if i == max_negatives: # Let's reconsider this
            break

    #return random.sample(negative_edges, int(len(negative_edges)*percentuale_negativi))
    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges


def shortest_path_based_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int,int,any]],type_src,type_dst,percentuale_negativi = 1.0,max_distance: int =10) -> set[tuple[int, int]]:

    """
    Generates negative edges by sampling node pairs that are either far apart in the graph
    (based on shortest path length) or belong to disconnected components, excluding existing edges.

    Parameters:
        g (nx.Graph):
            The graph containing nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            List of positive edges as tuples (source, destination, label).
            Label is ignored.

        max_distance (int):
            Minimum shortest path distance threshold for negative edge sampling
            (default=10).

    Returns:
        set (set[tuple[int, int]]):
            Set of predicted negative edges.

    Description:
        - Randomly selects pairs of distinct nodes not connected by an existing edge
        - Computes shortest path length between the nodes
        - Adds the pair as a negative edge if:
            1. The shortest path length is greater than or equal to the specified threshold, or
            2. No path exists between the nodes (disconnected components)
        - Returns when reaching target count or after max iterations
    """
    random.seed(42)

    negative_edges = set()

    # Find all pairs of nodes in the graph
    nodes = list(g.nodes())


    max_negatives = len(pos_edges)*3

    pbar = tqdm(desc = "Predicting negatives: ",total=max_negatives, dynamic_ncols=True)
    i = 0

    while len(negative_edges) < max_negatives:
        # Select two random nodes
        u, v = random.sample(nodes, 2)

        # Avoid edges between already-connected nodes
        if u != v and g.nodes[u]["label"] == type_src and g.nodes[v]["label"] == type_dst and not g.has_edge(u, v):
            try:
                # Compute the shortest distance between u and v
                dist = nx.shortest_path_length(g, source=u, target=v)

                # If the distance between u and v is greater than the threshold (or if they belong to separate components)
                if dist >= max_distance:
                    negative_edges.add((u, v))
            except nx.NetworkXNoPath:
                # This happens when there is no path between u and v (i.e. they are in separate components)
                negative_edges.add((u, v))
                i += 1


        pbar.update(1)
        if i == max_negatives:
            print(f'Number of predicted negatives: {i}') # Let's reconsider this
            break

    #return random.sample(negative_edges, int(len(negative_edges)*percentuale_negativi))
    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges



def pagerank_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int,int,any]],percentuale_negativi = 1.0) -> set[tuple[int, int]]:

    """
    Generates negative edges by sampling node pairs from the top-ranked nodes according to PageRank algorithm,
    excluding existing edges.

    Parameters:
        g (nx.Graph):
            The graph containing nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            List of positive edges as tuples (source, destination, label).
            Label is ignored.


    Returns:
        set (set[tuple[int, int]]):
            Set of predicted negative edges.

    Description:
        - Computes PageRank scores for all nodes in the graph
        - Selects the top 33% nodes based on PageRank scores
        - Samples node pairs from these top-ranked nodes ensuring:
            1. Nodes are distinct
            2. No existing edge between the pair
        - Returns when reaching target count or after max iterations
    """
    random.seed(42)

    # Compute PageRank
    pagerank_scores = nx.pagerank(g)

    # Sort nodes by descending PageRank
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:len(sorted_nodes) // 3]]  # Take the top 33% of nodes

    negative_edges = set()
    i = 0

    max_negatives = len(pos_edges)*3

    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    while len(negative_edges) < max_negatives:
        u, v = random.sample(top_nodes, 2)  # Take two nodes with high PageRank

        if u != v:
            negative_edges.add((u, v))

        i += 1
        pbar.update(1)
        if i == max_negatives:  # Let's reconsider this
            break

    #return random.sample(negative_edges, int(len(negative_edges)*percentuale_negativi))
    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges


def personalized_pagerank_negative_sampling(g: nx.Graph, pos_edges: list[tuple[int,int,any]],percentuale_negativi = 1.0) -> set[tuple[int, int]]:

    """
    Generates negative edges by sampling node pairs from the top-ranked nodes according to Personalized PageRank,
    personalized on nodes involved in positive edges, excluding existing edges.

    Parameters:
        g (nx.Graph):
            The graph containing nodes and edges.

        pos_edges (list[tuple[int, int, Any]]):
            List of positive edges as tuples (source, destination, label).
            Label is ignored.


    Returns:
        set (set[tuple[int, int]]):
            Set of predicted negative edges.

    Description:
        - Extracts nodes involved in positive edges
        - Computes Personalized PageRank with personalization vector focused on nodes involved in positive edges
        - Selects the top 33% nodes based on Personalized PageRank scores
        - Samples node pairs from these top-ranked nodes ensuring:
            1. Nodes are distinct
            2. No existing edge between the pair
        - Returns when reaching target count or after max iterations
    """
    random.seed(42)
    # Get the nodes involved in positive edges
    seed_nodes = {u for u, v, _ in pos_edges} | {v for u, v, _ in pos_edges}

    # Compute the Personalized PageRank
    ppr = nx.pagerank(g, personalization={node: 1 for node in seed_nodes})

    # Sort nodes by Personalized PageRank
    sorted_nodes = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:len(sorted_nodes) // 3]]

    negative_edges = set()
    i = 0

    max_negatives = len(pos_edges)*3

    pbar = tqdm(total=max_negatives, dynamic_ncols=True)

    while len(negative_edges) < max_negatives:
        u, v = random.sample(top_nodes, 2)
        if u != v:
            negative_edges.add((u, v))

        i += 1
        pbar.update(1)
        if i == max_negatives:  # Let's reconsider this
            break

    #return random.sample(negative_edges, int(len(negative_edges)*percentuale_negativi))
    pbar.close()
    return negative_edges
    #return random.sample(negative_edges, number_of_negatives) if  len(negative_edges) > number_of_negatives else negative_edges
