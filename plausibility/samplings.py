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

    # Raggruppiamo gli archi in base al peso
    for edge in edges:
        node1, node2, weight = edge  # Supponiamo che ogni edge sia (n1, n2, peso)
        edges_by_weight[weight].append(edge)

    sampled_edges = []
    for weight, edge_list in edges_by_weight.items():
        sample_size = int(len(edge_list) * sample_fraction)
        sampled_edges.extend(random.sample(edge_list, max(1, sample_size)))  # Manteniamo almeno 1 arco per peso

    return sampled_edges


def sampling_graph_with_distribution_edges(graph,pos_edges,sample_fraction, real_kg = False, neg_edges = [],seed = 42):

    #Se non uso un kg reale allora avrà un grafo orientato semplice (cioè che non ammette self loop e archi multipli), altrimenti avrò un grafo orientato che li accetterà
    new_graph = nx.DiGraph() if not real_kg else nx.MultiDiGraph()
    #Mappa nodo -> grado di uscita 
    out_degrees = dict(graph.out_degree()) # grado di uscita di ogni nodo del grafo
    #Per riproducibilità
    rng = np.random.default_rng(seed)
    #Somma di tutti gradi di uscita di tutti i nodi del grafo
    total = sum(out_degrees.values())

    total_edges = int(len(graph.edges())*sample_fraction) #51k
    i = 0
    #Lista archi campionati per non campionare gli stessi archi
    pos_edges_extracted = []
    neg_edges_extracted = []
    
    sample_nodes_probability = [(node,out_degrees[node] / total) for node in graph.nodes()]
    while i < total_edges:
        #NOTE: campiono un nodo alla volta in base al suo grado. 
        sampled_node = rng.choice([node[0] for node in sample_nodes_probability],p = [node[1] for node in sample_nodes_probability], replace = True, size = 1)
        #Prendo archi uscenti del nodo campionato.
        out_edges = graph.out_edges(sampled_node, data = True)
        if not real_kg:
            # <= 0 --> neg; >0 -->pos
            #Prendo archi uscenti positivi dal nodo campionato
            out_positivies_edges = [x for x in out_edges if int(x[2]['label']) > 0]
            #Prendo archi uscenti negativi dal nodo campionato
            out_negatives_edges = [x for x in out_edges if int(x[2]['label']) <= 0]

            #Probabiltà di estrarre un arco positivo o negativo
            edge_probability = [len(out_positivies_edges)/len(out_edges),len(out_negatives_edges)/len(out_edges)]

            #NOTE: estraggo due valori mantenendo la distribuzione degli archi negativi e positivi uscenti di un nodo:
            #se esce 1 allora pesco dal gruppo positivo, -1 pesco dal gruppo dei negativi, ovviamente le probabilità di 1 e -1 cambiamo in base a quanti positivi e negativi
            # escono dal nodo estratto
            # 1 = estraggo arco positivo, -1 = estraggo arco negativo
            label_sampled_edge = rng.choice([1,-1], p = edge_probability, size  = 1)


            if label_sampled_edge == 1:
                #NOTE: se estraggo positivo
                sampled_outer_edge = rng.choice(out_positivies_edges,size = 1)[0]
                sampled_outer_edge = (sampled_outer_edge[0],sampled_outer_edge[1],sampled_outer_edge[2]['label'])
                if sampled_outer_edge in pos_edges and sampled_outer_edge not in pos_edges_extracted:
                    pos_edges_extracted.append(sampled_outer_edge)
                    i += 1
            else:
                #NOTE: se non ho estratto positvo, ho estratto un negativo
                sampled_outer_edge = rng.choice(out_negatives_edges,size = 1)[0]
                sampled_outer_edge = (sampled_outer_edge[0],sampled_outer_edge[1],sampled_outer_edge[2]['label'])
                if sampled_outer_edge in neg_edges and sampled_outer_edge not in neg_edges_extracted:
                    neg_edges_extracted.append(sampled_outer_edge)
                    i += 1
        else:
            #Se ho un vero kg, ho solo archi positivi dunque ne campiono uno in modo uniforme, se non è già stato campionato lo aggiungo alla lista degli archi campionati
            sampled_outer_edge = rng.choice(list(out_edges),size = 1)[0]
            edge = (sampled_outer_edge[0],sampled_outer_edge[1])
            if edge not in pos_edges_extracted:
                pos_edges_extracted.append(edge)
                i += 1
                
    #Se non ho un vero kg avrò campionato sia archi positivi che negativi, alrimenti solo archi positivi
    edge_extracted = np.vstack((pos_edges_extracted,neg_edges_extracted)) if not real_kg else pos_edges_extracted

    #Ricostruisco il grafo
    for edge in edge_extracted:
        node_src, node_dst, peso = edge
        new_graph.add_edges_from([(node_src,node_dst,{'label' : peso})])

    return new_graph,pos_edges_extracted,neg_edges_extracted