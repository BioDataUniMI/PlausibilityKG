import grape
from grape.embedders import TransEEnsmallen, Node2VecSkipGramEnsmallen
from embiggen.embedders.pykeen_embedders.complex import ComplExPyKEEN
from embiggen.embedders.pykeen_embedders.rotate import RotatEPyKEEN
from embiggen.embedders.pykeen_embedders.transh import TransHPyKEEN
from embiggen.embedders.pykeen_embedders.distmult import DistMultPyKEEN
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import numpy as np
import pandas as pd

# TODO: aggiungere un po di documentazione mantenendo lo stesso pattern
# TODO: sistemare segnature delle funzioni 

def grape_graph_from_networkx(graph: nx.DiGraph, dataset_name: str, real_kg_graph = False, undirected = False) -> grape.Graph:

    """
    Convert a Networkx Graph in a Grape Graph.
    
    Parameters:

        graph (nx.DiGraph) : 
            A Networkx Graph to convert.
        real_kg_bool (bool): 
            If is True, that means that we are using a real KG graph, otherwise not.
        dataset_name (str) : 
            The name of the new Grape Graph.

    Returns:

        graph (grape.Graph):
            The Grape Graph based on the Networkx graph.

    """

    if real_kg_graph:
        list_edges = [(u,v,predicate['label']) for u,v,predicate in list(graph.edges(data = True))]
        edges = pd.DataFrame(list_edges, columns=["subject", "object","predicate"])
        #NOTE: rileggo tutti i nodi per riprendere il tipo dei nodi 
        #nodes = pd.read_csv(f'data/nodes_{dataset_name}.tsv', sep = '\t')
        #NOTE: al momento non trovo altro modo che far così, perchè non penso che networkx mi permetta di avere nodi del tipo (nome,tipo) e archi (nome_src,nome_dst,predicate), 
        # ma se ho nodi del tipo (nome,tipo) penso che mi permetta solo di avere archi del tipo ((nome_src, tipo),(nome_dst,tipo),predicate)
        #list_all_nodes = list(zip(nodes['name'],nodes['type'])) # list of nodes
        #NOTE: il mapping di nodi non presenti nel grafo viene ignorato, faccio in questo modo così evito di aggiungere tutti i nodi e poi togliere quelli disconnessi
        #mapping = {n[0]: (n[0],n[1]) for n in list_all_nodes}
        #nx.relabel_nodes(graph, mapping, copy=False)
        list_graph_nodes = [(n,type['label']) for n, type in list(graph.nodes(data = True))]
        nodes = pd.DataFrame(list_graph_nodes, columns = ["name","type"])
    else:
        #NOTE: '*' simbolo di default per predicate e tipo dei nodi se usiamo i grafi presi dai dataset
        list_edges = [(u,v,'A') for u, v in list(graph.edges())] # ignoriamo predicato dato che facciamo embedding solo dei nodi
        edges = pd.DataFrame(list_edges, columns=["subject", "object","predicate"])
        print(list(graph.nodes(data = True))[:10])
        list_nodes = [(u,type['label']) for u, type in list(graph.nodes(data = True))] #NOTE: per transe serve il tipo del nodo pure
        nodes = pd.DataFrame(list_nodes, columns = ["name","type"])
        #print(nodes[:10])

    directed = False if undirected else True

    graph = grape.Graph.from_pd(
        edges_df=edges,
        nodes_df=nodes,
        node_name_column="name",
        node_type_column="type",
        node_types_separator=";",   
        edge_src_column="subject",
        edge_dst_column="object",
        edge_type_column = "predicate",
        directed= directed,
        name=dataset_name
    )

    print(f'Grape graph ha: {graph.get_number_of_nodes()} nodi e {graph.get_number_of_edges()} archi')

    return graph

def complex_to_real(arr: np.ndarray) -> np.ndarray:
    """
        Convert a complex array to interleaved real/imaginary floats.
    """
    return np.stack([arr.real, arr.imag], axis=1).flatten()


def apply_transe_embedding(graph: grape.Graph):
    """
    Apply transE embedding on a Grape graph

    Paramaters:

        graph (grape.Graph):
            A grape Graph 

    Returns:

        graph_embedding_transe (grape.EmbeddingResult):
            Contains the embeddings 
    """



    print("Using transE Embedding")
    embedder_transE = TransEEnsmallen(random_state=42,embedding_size = 32)
    graph_embedding_transe = embedder_transE.fit_transform(graph)



    return graph_embedding_transe

def apply_complex_embedding(graph: grape.Graph):
    """
    Apply ComplEx embedding on a Grape graph.

    Parameters
    ----------
    graph : grape.Graph

    Returns
    -------
    grape.EmbeddingResult
    """
    print("Using ComplEx Embedding")
    embedder_complex = ComplExPyKEEN(random_state=42, embedding_size=16)
    return embedder_complex.fit_transform(graph)


def apply_rotate_embedding(graph: grape.Graph):
    """
    Apply RotatE embedding on a Grape graph.

    Parameters
    ----------
    graph : grape.Graph

    Returns
    -------
    grape.EmbeddingResult
    """
    print("Using RotatE Embedding")
    embedder_rotate = RotatEPyKEEN(random_state=42, embedding_size=16)
    return embedder_rotate.fit_transform(graph)


def apply_transh_embedding(graph: grape.Graph):
    """
    Apply TransH embedding on a Grape graph.

    Parameters
    ----------
    graph : grape.Graph

    Returns
    -------
    grape.EmbeddingResult
    """
    print("Using TransH Embedding")
    embedder_transh = TransHPyKEEN(random_state=42, embedding_size=32)
    return embedder_transh.fit_transform(graph)

def apply_distmult_embedding(graph: grape.Graph):
    """
    Apply DistMult embedding on a Grape graph.

    Parameters
    ----------
    graph : grape.Graph

    Returns
    -------
    grape.EmbeddingResult
    """
    print("Using DistMult Embedding")
    embedder_distmult = DistMultPyKEEN(random_state=42, embedding_size=32)
    return embedder_distmult.fit_transform(graph)

def mapping_transe_embedding_to_edge(positives: list[tuple[any,any,any]], predicted_negatives: list[tuple[any,any]],embedding_nodes,embedding_type_edges, real_kg_graph: bool):
    """
    Create a dictionary that has the embedding of an edge as a key and the edge corrisponded to the embedding as a value

    Paramaters:

        positives (list[tuple[any,any,any]]):
            List of positives typed edges 

        negatives (list[tuple[any,any]]):
            List of predicted negatives not typed edges
        
        embedding_nodes ():
            The nodes embeddings

        embedding_type_edges ():
            The type edges embedding

        real_kg_bool (bool): 
            If is True, that means that we are using a real KG graph, otherwise not.

    Returns:

        embedding_to_edge (dict):
            Mapping each embedding edge to the corrisponded edge 
        
        X_pos (List[tuple]):
            List that contains the embeddings of positives edges

        X_neg (list[tuple]):
            List that contains the embeddings of predicted negatives
    """
        
    embedding_to_edge = {}
    X_pos = []
    X_neg = []
    df_embedding_nodes = embedding_nodes[0]
    df_embedding_type_edges = embedding_type_edges[0] if real_kg_graph else None

    for p in positives:
        emb_src = df_embedding_nodes.loc[p[0]]
        emb_dst = df_embedding_nodes.loc[p[1]]
        emb_edge = np.multiply(emb_src.tolist(),emb_dst.tolist())
        X_pos.append(emb_edge)
        embedding_to_edge[tuple(emb_edge.tolist())] = p  # Salva la mappatura (embedding → arco)

    print(f'Mapped {len(embedding_to_edge.keys())} positives edges')


    for n in (predicted_negatives):
        emb_src = df_embedding_nodes.loc[n[0]]
        emb_dst = df_embedding_nodes.loc[n[1]]
        emb_edge = np.multiply(emb_src.tolist(),emb_dst.tolist())
        X_neg.append(emb_edge)
        embedding_to_edge[tuple(emb_edge.tolist())] = n  # Salva la mappatura (embedding → arco) # Salva la mappatura (embedding → arco) 


    return embedding_to_edge, X_pos, X_neg

def store_embedding(embedding: grape.EmbeddingResult, folder: str, filename: str, embedding_name : str,real_kg_graph = False):
    """
    Store nodes and types edges embeddings in a csv file.

    Paramaters:

        embedding ():
            Contains the embeddings of the Grape graph.

        folder (str):
            Name of the folder where we want to store the embeddings.
        
        filename (str):
            Name of the csv file where we want to store the embeddings.

        real_kg_graph(bool):
            If it is True means that we are using a real kg, otherwise not.

    """
    emb_i = embedding.get_node_embedding_from_index(0)
    emb_i.loc[:, 'embedding'] = emb_i.iloc[:, 0:].apply(lambda row: row.to_list(), axis=1)#iloc: tutte le righe e tutte le colonne tranne l'indice
    emb_i.index.name = 'name'
    file = f'store_embeddings/{embedding_name}/{filename}.csv'
    emb_i[['embedding']].to_csv(file, sep=',')
    print(f'Embedding salvati in: {file}')

    if real_kg_graph:
        #NOTE: se abbiamo un kg vero abbiamo i predicate sugli archi e dunque ne salvo gli embedding
        emb_type_edge_i = embedding.get_edge_type_embedding_from_index(0)
        emb_type_edge_i.loc[:, 'embedding'] = emb_type_edge_i.iloc[:, 0:].apply(lambda row: row.to_list(), axis=1)
        emb_type_edge_i.index.name = 'predicate'
        emb_type_edge_i[['embedding']].to_csv(folder + f'{filename}_{embedding_name}_types_edges_embedding.csv', sep=',')


def apply_node2vec_embedding(graph: nx.DiGraph) -> HadamardEmbedder: 
    """
    Apply node2vec embedding on a Networkx graph

    Paramaters:

        graph (nx.DiGraph):
            A Networkx Graph 

    Returns:

        edge_embs (HadamardEmbedder):
            Contains the embeddings 
    """
    print("Using Node2Vec Embedding")

    embedder_node2vec = Node2VecSkipGramEnsmallen(random_state=42, embedding_size=100)
    graph_embedding_node2vec = embedder_node2vec.fit_transform(graph) # NB. undirected graph
    return graph_embedding_node2vec


def mapping_node2vec_embedding_to_edge(positives: list[tuple[any,any,any]], predicted_negatives: list[tuple[any,any]],embedding_nodes):
    """
    Create a dictionary that has the embedding of an edge as a key and the edge corrisponded to the embedding as a value

    Paramaters:

        positives (list[tuple[any,any,any]]):
            List of positives edges 

        negatives (list[tuple[any,any]]):
            List of predicted negatives edges
        
        embedding_nodes (HadamardEmbedder):
            The nodes embeddings

        predicted_negatives (list[tuple[any,any]]):
            List of predicted edges negatives

        use_predicted_negatives (bool,optional): 
            If is True, that means that we are using the predicted negatives edges for training and evaluating.

    Returns:

        embedding_to_edge ():
            Mapping each embedding edge to the corrisponded edge 
        
        X_pos (list[tuple]):
            List that contains embeddings of positives edges

        X_neg (list[tuple]):
            List that contanins embeddings of predicted negatives edges if use_predicted_negatives is True, otherwise contains embeddings of real negatives edges
    """
        
    embedding_to_edge = {}
    X_pos = []
    X_neg = []
    df_embedding_nodes = embedding_nodes[0]

    for p in positives:
        emb_src = df_embedding_nodes.loc[p[0]]
        emb_dst = df_embedding_nodes.loc[p[1]]
        emb_edge = np.concatenate([emb_src.tolist(),emb_dst.tolist()])
        X_pos.append(emb_edge)
        embedding_to_edge[tuple(emb_edge.tolist())] = p  # Salva la mappatura (embedding → arco)

    


    #NOTE: per evitare il cambiamento nella funzione train_evaluate, dovrei aggiunge ai predetti negativi una label 'fittizia' in modo tale che l'arco sia composto da 3 elementi e non debba fare
    # più il controllo sulla lunghezza dell'arco 
    for n in (predicted_negatives):
        emb_src = df_embedding_nodes.loc[n[0]]
        emb_dst = df_embedding_nodes.loc[n[1]]
        emb_edge = np.concatenate([emb_src.tolist(),emb_dst.tolist()])
        X_neg.append(emb_edge)
        embedding_to_edge[tuple(emb_edge.tolist())] = n  # Salva la mappatura (embedding → arco) # Salva la mappatura (embedding → arco) 


    return embedding_to_edge, X_pos, X_neg