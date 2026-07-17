from collections import defaultdict
import networkx as nx
import numpy as np
import random


def stratified_sampling_by_weight(edges, sample_fraction):
    """
    Perform stratified sampling on edges, keeping the proportion of edges per weight category.
    """
    random.seed(42)

    edges_by_weight = defaultdict(list)

    # Group the edges by weight
    for edge in edges:
        node1, node2, weight = edge  # Assume each edge is (n1, n2, weight)
        edges_by_weight[weight].append(edge)

    sampled_edges = []
    for weight, edge_list in edges_by_weight.items():
        sample_size = int(len(edge_list) * sample_fraction)
        sampled_edges.extend(random.sample(edge_list, max(1, sample_size)))  # Keep at least 1 edge per weight

    return sampled_edges


def sampling_graph_with_distribution_edges(graph,pos_edges,sample_fraction, real_kg = False, neg_edges = [],seed = 42):

    #If not using a real KG, use a simple directed graph (i.e. no self loops or multi-edges), otherwise use a directed graph that allows them
    new_graph = nx.DiGraph() if not real_kg else nx.MultiDiGraph()
    #Map node -> out-degree
    out_degrees = dict(graph.out_degree()) # out-degree of each node in the graph
    #For reproducibility
    rng = np.random.default_rng(seed)
    #Sum of the out-degrees of all nodes in the graph
    total = sum(out_degrees.values())

    total_edges = int(len(graph.edges())*sample_fraction) #51k
    i = 0
    #List of sampled edges, to avoid sampling the same edge twice
    pos_edges_extracted = []
    neg_edges_extracted = []

    sample_nodes_probability = [(node,out_degrees[node] / total) for node in graph.nodes()]
    while i < total_edges:
        #NOTE: sample one node at a time based on its degree.
        sampled_node = rng.choice([node[0] for node in sample_nodes_probability],p = [node[1] for node in sample_nodes_probability], replace = True, size = 1)
        #Take the outgoing edges of the sampled node.
        out_edges = graph.out_edges(sampled_node, data = True)
        if not real_kg:
            # <= 0 --> neg; >0 -->pos
            #Take the positive outgoing edges from the sampled node
            out_positivies_edges = [x for x in out_edges if int(x[2]['label']) > 0]
            #Take the negative outgoing edges from the sampled node
            out_negatives_edges = [x for x in out_edges if int(x[2]['label']) <= 0]

            #Probability of drawing a positive or negative edge
            edge_probability = [len(out_positivies_edges)/len(out_edges),len(out_negatives_edges)/len(out_edges)]

            #NOTE: draw one of two values keeping the distribution of the node's outgoing negative and positive edges:
            #if 1 comes up, sample from the positive group; if -1, sample from the negative group - the probabilities of 1 and -1 obviously change based on how many positives and negatives
            # come out of the sampled node
            # 1 = draw a positive edge, -1 = draw a negative edge
            label_sampled_edge = rng.choice([1,-1], p = edge_probability, size  = 1)


            if label_sampled_edge == 1:
                #NOTE: if a positive edge is drawn
                sampled_outer_edge = rng.choice(out_positivies_edges,size = 1)[0]
                sampled_outer_edge = (sampled_outer_edge[0],sampled_outer_edge[1],sampled_outer_edge[2]['label'])
                if sampled_outer_edge in pos_edges and sampled_outer_edge not in pos_edges_extracted:
                    pos_edges_extracted.append(sampled_outer_edge)
                    i += 1
            else:
                #NOTE: if a positive wasn't drawn, a negative was drawn
                sampled_outer_edge = rng.choice(out_negatives_edges,size = 1)[0]
                sampled_outer_edge = (sampled_outer_edge[0],sampled_outer_edge[1],sampled_outer_edge[2]['label'])
                if sampled_outer_edge in neg_edges and sampled_outer_edge not in neg_edges_extracted:
                    neg_edges_extracted.append(sampled_outer_edge)
                    i += 1
        else:
            #If using a real KG, there are only positive edges, so sample one uniformly; if it hasn't already been sampled, add it to the list of sampled edges
            sampled_outer_edge = rng.choice(list(out_edges),size = 1)[0]
            edge = (sampled_outer_edge[0],sampled_outer_edge[1])
            if edge not in pos_edges_extracted:
                pos_edges_extracted.append(edge)
                i += 1

    #If not using a real KG, both positive and negative edges will have been sampled; otherwise only positive edges
    edge_extracted = np.vstack((pos_edges_extracted,neg_edges_extracted)) if not real_kg else pos_edges_extracted

    #Rebuild the graph
    for edge in edge_extracted:
        node_src, node_dst, peso = edge
        new_graph.add_edges_from([(node_src,node_dst,{'label' : peso})])

    return new_graph,pos_edges_extracted,neg_edges_extracted
