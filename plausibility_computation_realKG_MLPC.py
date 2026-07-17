import ast
import grape
from grape.embedders import TransEEnsmallen
from ensmallen import Graph
import csv
import itertools as it
import logging
from joblib import Parallel, delayed
import joblib
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import numpy as np
import os
import pandas as pd
import pickle
import re
import time
import json
from tqdm import tqdm
from typing import Callable
from print_color import print

from plausibility.classifiers import OCSVM
from plausibility.dataset import download_dataset, read_dataset
from plausibility.embedding_utils import *
from plausibility.metrics import *
from plausibility.samplings import *
from plausibility.strategies import *
from plausibility.visualizations import *

import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef



def import_data(name_view_graph: str):

    edges = pd.read_csv(f'data/edges_{name_view_graph}.csv',sep = ',',low_memory = False)
    nodes = pd.read_csv(f'data/nodes_{name_view_graph}.csv',sep = ',', low_memory = False)

    nodes = nodes[[':LABEL','URI:ID']]
    edges = edges[[':START_ID', ':TYPE', ':END_ID']]

    return edges,nodes



def node_type_join(edge_df: pd.DataFrame, node_df: pd.DataFrame) -> pd.DataFrame:

    # Add subject type to edge rows
    subject_merge = edge_df.merge(node_df, left_on=':START_ID', right_on='URI:ID') \
                        .drop(['URI:ID'], axis = 1) \
                        .rename(columns={':LABEL': 'START_ID_LABEL'})
    # Add object type to edge rows
    subject_object_merge = subject_merge.merge(node_df, left_on=':END_ID', right_on='URI:ID') \
                                        .drop(['URI:ID'], axis = 1) \
                                        .rename(columns={':LABEL': 'END_ID_LABEL'})


    return subject_object_merge



def filter_edge_type(types: str, type_df: pd.DataFrame) -> pd.DataFrame:
    # Take types and change data structure
    types = types.split(',')
    types = [t.split(' - ') for t in types]
    types = [[t[0], t[1].replace(' ', '_'), t[2]] for t in types]
    key = ""

    edge_type_dict = {}
    for t in types:
        key = t[0] + '-' + t[1] + '-' + t[2]
        edge_type_dict[key] = type_df[( type_df['START_ID_LABEL'].str.split(';').apply(lambda x: t[0] in x)) & ( type_df[':TYPE'] == t[1]) & (type_df['END_ID_LABEL'].str.split(';').apply(lambda x: t[2] in x))]

    
    return edge_type_dict, key


def create_rnakg_graph(edges_df: pd.DataFrame,name_view_graph: str) -> nx.digraph:

    edges = list(zip(edges_df[':START_ID'],edges_df['START_ID_LABEL'],edges_df[':END_ID'],edges_df['END_ID_LABEL'],edges_df[":TYPE"])) # list of edges

    g = nx.MultiDiGraph()

    for source, type_src, dst, type_dst, predicate in tqdm(edges, desc = f'Creating RNA-KG ({name_view_graph} view)'):
        g.add_node(source, label=type_src)
        g.add_node(dst, label=type_dst)
        g.add_edges_from([(source,dst)],label = predicate)

    return g



def create_subgraph(filtered_edges,relation: str):
    edges = filtered_edges[relation]
    subgraph = nx.MultiDiGraph()
    occ_type = {}

    type_source = relation.split("-")[0]
    type_target = relation.split("-")[2]
    occ_type[type_source] = 0 
    occ_type[type_target] = 0
    


    for i in tqdm(range(len(edges)), desc = "Creating Subgraph"):
        row = edges.iloc[i].tolist()

        if len(row[3]) > 1: 
            if type_source in row[3].split(";"): type_src = type_source
        else:
            type_src = row[3]

        if len(row[4]) > 1: 
            if type_target in row[4].split(";"): type_dst = type_target
        else:
            type_dst = row[4]


        if row[0] not in list(subgraph.nodes()): occ_type[type_src] += 1
        if row[2] not in list(subgraph.nodes()): occ_type[type_dst] += 1

        subgraph.add_node(row[0], label=type_src)
        subgraph.add_node(row[2], label=type_dst)

        subgraph.add_edge(row[0], row[2], label=row[1])

    number_type_source = occ_type[type_source]
    number_type_target = occ_type[type_target]
    
    return subgraph, number_type_source, number_type_target


def apply_node2vec(nx_graph: nx.MultiDiGraph, path = ''):
    
    node2vec = Node2Vec(nx_graph, dimensions=32, walk_length=30, num_walks=200, workers=16,seed = 42)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    if path:
        rows = []
        for node_id in model.wv.index_to_key:
            embedding = model.wv[node_id].tolist()
            rows.append((node_id, embedding))

        # Crea DataFrame
        df = pd.DataFrame(rows, columns=["node_id", "embedding"])

        # Salva su CSV
        df.to_csv(path, index=False)
        print(f"Embeddings salvati in {path}")

    return edges_embs


def apply_strategy(graph: nx.MultiDiGraph, strategy ,positives,type_src,type_dst,load_pred_negs = False):
    if not load_pred_negs:
        predicted_negatives = strategy(graph, positives,type_src,type_dst) # Predice dei possibili negativi         
    return predicted_negatives


def compute_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, specificity, and F-beta score.
    """

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()

    accuracy = (tp + tn) / (tp + fp + fn + tn) 
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    fnr = fn / (fn + tp) if fn + tp > 0 else np.nan
    fpr = fp / (tn + fp) if tn + fp > 0 else np.nan

    f_beta_1 = F_beta_score(precision, recall, beta = 1)
    f_beta_2 = F_beta_score(precision, recall, beta = 2)
    error_beta_score_0_5 = error_beta_score(fpr,fnr,beta = 0.5)
    error_beta_score_1 = error_beta_score(fpr,fnr,beta = 1)
    error_beta_score_2 = error_beta_score(fpr,fnr,beta = 2)

    print(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn},acc: {accuracy}, precision: {precision}, recall: {recall}, specificity: {specificity}, fpr: {fpr}, fnr: {fnr}, f_beta_1: {f_beta_1}, f_beta_2: {f_beta_2}, error_beta_score_0_5: {error_beta_score_0_5}, error_beta_score_1: {error_beta_score_1}, error_beta_score_2: {error_beta_score_2}")

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "fpr": fpr, "fnr": fnr, "f_beta_1": f_beta_1, "f_beta_2": f_beta_2, "error_beta_score_0_5": error_beta_score_0_5, "error_beta_score_1": error_beta_score_1, "error_beta_score_2": error_beta_score_2}


def evaluate_hyperparams(alpha, hidden_layer, X_trainval, y_trainval, beta):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner_scores = []

    for train_idx, val_idx in inner_cv.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                alpha=alpha,
                hidden_layer_sizes=hidden_layer,
                max_iter=20000,
                tol=1e-4,
                random_state=42,
                early_stopping=True
            )
        )

        model.fit(X_train, y_train)
        y_hat = model.predict(X_val)

        tn, fp, fn, tp = metrics.confusion_matrix(y_val, y_hat, labels=[-1,1]).ravel()
        fpr = fp/(tn+fp) if (tn+fp)>0 else 0.0
        fnr = fn/(fn+tp) if (fn+tp)>0 else 1.0
        score = error_beta_score(fpr, fnr, beta)   # resta coerente col tuo obiettivo
        inner_scores.append(score)

    return alpha, hidden_layer, float(np.mean(inner_scores))

#*----

def find_best_hyperparams(X_trainval, y_trainval, alpha_values, hidden_layer_values, beta):
    """
    Performs grid search over nu and gamma values using parallel processing.
    Returns the best hyperparameters.
    """
    search_results = Parallel(n_jobs=-1)(
        delayed(evaluate_hyperparams)(alpha, hidden_layer, X_trainval, y_trainval, beta)
        for alpha, hidden_layer in it.product(alpha_values, hidden_layer_values)
    )

    return min(search_results, key=lambda x: x[2])  # Best alpha, hidden_layer


def calculate_mean_std(results):
    metric_names = ["accuracy","precision","recall","specificity","fpr","fnr",
                    "f_beta_1","f_beta_2","error_beta_score_0_5",
                    "error_beta_score_1","error_beta_score_2", "balanced_accuracy", "MCC"]
    summary = {}
    for m in metric_names:
        arr = np.array([r[m] for r in results], dtype=float)
        arr = arr[np.isfinite(arr)]
        summary[m] = "N/A" if arr.size == 0 else f"{arr.mean():.3f}\\pm{arr.std(ddof=1):.3f}"
    return summary

# def calculate_mean_std(results):
#     metrics = ["accuracy", "precision", "recall", "specificity","fpr","fnr",
#                "f_beta_1","f_beta_2","error_beta_score_0_5",
#                "error_beta_score_1","error_beta_score_2", "MCC"]
#     summary = {}

#     for metric in metrics:
#         values = [res[metric] for res in results if not np.isnan(res[metric])]
#         if np.nan not in values and len(values) > 0:
#             mean = np.mean(values)
#             std_dev = np.std(values)
#             summary[metric] = ("{:.3f}{}pm{:.3f}").format(mean, "\\",std_dev)

#         else:
#             summary[metric] = "N/A"

#     return summary


def train_and_evaluate(X, y, alpha_values, hidden_layer_values, embedding_to_edge, relation: str,strategy_name : str, name_view_graph: str, embedding_name: str, model_selection_parameter: str, dump: bool ,serialize_models = False,  save_serialized_models_dir = '', 
                       save_drawings_path = '', draw_graph = False, combined_negatives = False): 
    """
    Trains and evaluates models using 5-fold cross-validation.
    Serializes models if enabled.
    """
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_params = {}
    metrics_results_list = [] 
    best_parameter_model_selection = 1.0 if model_selection_parameter == "error_beta_score_2" or model_selection_parameter == "error_beta_score_0_5" or model_selection_parameter == "error_beta_score_1" else 0.0 #NOTE: 1.0 se si vuole minimizzare il parametro, se si vuole massimizzare mettere 0.0
    total_tp, total_fp , total_tn , total_fn = 0,0,0,0

    beta = 0
    if model_selection_parameter == "error_beta_score_2": beta = 2.0
    elif model_selection_parameter == "error_beta_score_1": beta = 1.0
    elif model_selection_parameter == "error_beta_score_0_5": beta = 0.5

    if serialize_models and save_serialized_models_dir:
        os.makedirs(save_serialized_models_dir, exist_ok=True)

    tp_arr, fp_arr, tn_arr, fn_arr = [], [], [], []
    for fold, (trainval_idx, test_idx) in tqdm(enumerate(outer_cv.split(X, y), start=1), total=outer_cv.get_n_splits(X, y), desc = "Training, Validation and Test MLPC",leave = True,position = 0):
        #Creo il grafo vuoto per ogni fold
        graph = nx.DiGraph()

        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]

        # Hyperparameter tuning
        best_alpha, best_hidden_layer, _ = find_best_hyperparams(X_trainval, y_trainval, alpha_values, hidden_layer_values, beta)
        # best_params = {'alpha': best_alpha}

        # Train final model
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                alpha=best_alpha,
                hidden_layer_sizes=best_hidden_layer,
                max_iter=20000,
                tol=1e-4,
                random_state=42,
                early_stopping=True
            )
        )

        model.fit(X_trainval, y_trainval)

        # Serialize the trained model
        if serialize_models:
            model_path = os.path.join(save_serialized_models_dir, f"model_fold_{fold}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
                print(f"Model for fold {fold} saved at {model_path}")

        # Evaluate model performance
        y_pred = model.predict(X_test)

        """
        correct_edges = {}
        for i, Xi in enumerate(X_test):
            # print(f'Fold: {fold}, {edge_to_embedding[tuple(Xi)]}, Pred: {y_pred[i]}, Test: {y_test[i]}')
        
            correct_edges[embedding_to_edge[tuple(Xi)]] = (y_pred[i] == y_test[i], y_test[i])
            #NOTE: dato che uso sia i predetti negativi che i positivi gestisco il fatto di avere tuple di 3 elementi (src,pred,dst) in caso di kg vero  e 2 elementi (src,dst)
            edge = embedding_to_edge[tuple(Xi.tolist())]
            if len(edge) == 2:
                scr_node, dst_node = edge
                graph.add_edges_from([(scr_node,dst_node)])
            else:
                scr_node, dst_node, predicate  = edge
                graph.add_edges_from([(scr_node,dst_node)],label = predicate)

        #Grafico per ogni fold
        if draw_graph:
            draw_fold_graph(graph, correct_edges, fold, save_drawings_path=save_drawings_path)
        """
        
        metrics_results = compute_metrics(y_test, y_pred)

        total_tp += metrics_results["tp"]
        total_fp += metrics_results["fp"]
        total_tn += metrics_results["tn"]
        total_fn += metrics_results["fn"]

        tp_arr.append(metrics_results["tp"])
        fp_arr.append(metrics_results["fp"])
        tn_arr.append(metrics_results["tn"])
        fn_arr.append(metrics_results["fn"])

        mcc = matthews_corrcoef(y_test, y_pred)

        #NOTE: < if we use error beta score, cause we want to minimize it, with accuracy is > cause we want to maximize acc

        if model_selection_parameter == "error_beta_score_2" or model_selection_parameter == "error_beta_score_0_5" or model_selection_parameter == "error_beta_score_1":
            if metrics_results[model_selection_parameter] < best_parameter_model_selection:
                best_parameter_model_selection = metrics_results[model_selection_parameter]
                best_params = {'alpha': best_alpha, 'hidden_layer': best_hidden_layer} 

        elif model_selection_parameter == "accuracy":
            if metrics_results[model_selection_parameter] > best_parameter_model_selection:
                best_parameter_model_selection = metrics_results[model_selection_parameter]
                best_params = {'alpha': best_alpha, 'hidden_layer': best_hidden_layer} 

        else:
            raise ValueError("Parameter for model selection not valid")

        metrics_results_list.append({
            "fold": fold,
            "accuracy": float(f'{metrics_results["accuracy"]:.3f}'),
            "precision": float(f'{metrics_results["precision"]:.3f}'),
            "recall": float(f'{metrics_results["recall"]:.3f}'),
            "specificity": float(f'{metrics_results["specificity"]:.3f}'),
            "fpr": float(f'{metrics_results["fpr"]:.3f}'),
            "fnr": float(f'{metrics_results["fnr"]:.3f}'),
            "f_beta_1": float(f'{metrics_results["f_beta_1"]:.3f}'),
            "f_beta_2": float(f'{metrics_results["f_beta_2"]:.3f}'),
            "error_beta_score_0_5": float(f'{metrics_results["error_beta_score_0_5"]:.3f}'),
            "error_beta_score_1": float(f'{metrics_results["error_beta_score_1"]:.3f}'),
            "error_beta_score_2": float(f'{metrics_results["error_beta_score_2"]:.3f}'),
            "best_alpha": best_alpha,
            "best_hidden_layer": best_hidden_layer,
            "balanced_accuracy":float(f'{((metrics_results["recall"]+ metrics_results["specificity"])/2):.3f}'),
            "MCC": mcc
        })

    if dump: 

        final_model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                alpha=best_params['alpha'],
                hidden_layer_sizes=best_params['hidden_layer'],
                max_iter=20000,
                tol=1e-4,
                random_state=42,
                early_stopping=True
            )
        )

        final_model.fit(X, y)

        err_beta = None
        if model_selection_parameter == 'error_beta_score_1': err_beta = 'err_beta_1'
        elif model_selection_parameter == 'error_beta_score_2': err_beta = 'err_beta_2'
        elif model_selection_parameter == 'error_beta_score_0_5': err_beta = 'err_beta_0_5'
        # Dump del modello finale
        if combined_negatives:
            path_model = f'dumps_models_new/{name_view_graph}/{embedding_name}/combined_c_s_d/{err_beta}/MLPC/MLCP_model_{relation}_fixed.pkl'
        else:
            path_model = f'dumps_models_new/{name_view_graph}/{embedding_name}/{strategy_name}/{err_beta}/MLPC/MLCP_model_{relation}_fixed.pkl'
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        with open(path_model, 'wb') as f:
            pickle.dump(final_model ,f)

        print(f'Modello salvato in {path_model}')
    
    # Print results as a table
    #print_results_table(results)
    print(metrics_results_list)
    return metrics_results_list,total_tp,total_fp,total_tn,total_fn,tp_arr,fp_arr,tn_arr,fn_arr,best_params



def compute(relation: str,strategy,strategy_name: str,name_view_graph: str,embedding_name: str, parameter_model_selection = 'accuracy', blind_test = False,load_embedding = False,embedding_subgraph = False,dump = False, combined_negatives = False):

    """
    PARAMS:

    relation: the relation used to create the subgraph of the view

    strategy: strategy used to predicted negatives edges

    strategy_name: the name of the strategy e.g strategy: community based negative sampling --> strategy_name: c-b-n-s

    name_view_graph: the name of the rna_kg's view

    embedding_name: the name of embedding 'n2v' or 'transe'

    parameter model selection: the parameter used in the model selection (DEFAULT is "accuracy")

    dump: True if you wanto to dump the model, otherwise False (DEFAULT is False)

    load_embedding: True if you want to use stored embeddings, otherwise False (DEFAULT is False)

    embedding_subgraph: True if you want to calculate embeddings on the subgraph, otherwise False (DEFAULT is False)

    blind_test: True if you want to save the 10% of the positives edges of the subgraph for future blind tests. otherwise False (DEFAULT is False)
    
    
    """
    
    ping = time.time()
    results = {}

    #import and prepare data
    edges, nodes = import_data(name_view_graph)
    triple_type_df = node_type_join(edges, nodes)
    print(triple_type_df.columns)

    # Create che rna KG  
    graph = create_rnakg_graph(triple_type_df, name_view_graph)
    print(f'RNA-kg ({name_view_graph}) has: {len(graph.nodes())} nodes and {len(graph.edges())} edges')

    # Create the subgraph that include only edges that match the input relation
    filtered_edges, filter = filter_edge_type(relation, triple_type_df)
    print('Relation: ', filter, ',' , filtered_edges[filter].shape[0], 'triples')
    subgraph,number_type_src,numeber_type_dst = create_subgraph(filtered_edges,filter)
    print(f'The subgraph ({relation}) has {len(list(subgraph.nodes()))} nodes and {len(list(subgraph.edges()))} edges')

    #Get positives edges of the subgraph
    positives_subgraph = list(subgraph.edges())

    predicted_negatives = None
    if combined_negatives:
        comb_negs = pd.read_csv(f"negative_samples/{name_view_graph}/combine_c_s_d_proportion/{relation}.csv")
        predicted_negatives = set()
        for row in comb_negs[['source', 'target']].itertuples(index=False):
            predicted_negatives.add(( row[0], row[1] ))

        print('# combined negatives:', len(predicted_negatives), background='w', color='red')

    else:
        # Check if negative samples already exist
        if os.path.exists(f'negative_samples/{name_view_graph}/{strategy_name}/{relation}.csv'):
            load_negatives = pd.read_csv(f"negative_samples/{name_view_graph}/{strategy_name}/{relation}.csv")
            predicted_negatives = set()
            for row in load_negatives[['source', 'target']].itertuples(index=False):
                predicted_negatives.add(( row[0], row[1] ))

            print('# loaded negatives:', len(predicted_negatives), background='magenta', color='w')
        else:
            #Apply strategy on the subgraph
            type_src = relation.split('-')[0].strip()
            type_dst = relation.split('-')[2].strip()
            predicted_negatives = apply_strategy(subgraph,strategy,positives_subgraph,type_src,type_dst)
            print(type(predicted_negatives))
            predicate = relation.split(' - ')[1]
            negative_with_types = {(*t, predicate, type_src, type_dst) for t in predicted_negatives}
            data = list(negative_with_types)
            predicted_negatives_df = pd.DataFrame(data, columns=['source', 'target', 'predicate', 'source_type', 'target_type'])
            predicted_negatives_df.to_csv(f'negative_samples/{name_view_graph}/{strategy_name}/{relation}.csv', index=False)
            print(f'{len(predicted_negatives)} predicted negatives biased')
            #if no pred negs biased return 
            if len(predicted_negatives) == 0:
                return

    #Take the predicted nevatives unbiased (with not positive edges)
    predicted_negatives =set(predicted_negatives) - set((x[0],x[1]) for x in positives_subgraph) 
    #for u,v in predicted_negatives:
        #print(f'Tipo arco {subgraph.nodes[u]["label"]} -> {subgraph.nodes[v]["label"]}')
    #if no pred negs unbiased return 
    if len(predicted_negatives) == 0:
        return
    print(f'{len(predicted_negatives)} predicted negatives unbiased')


    if blind_test:
        #Get edges with predicate
        positives_subgraph = [(x[0],x[1],x[2]["label"]) for x in list(subgraph.edges(data = True))]
        #Take the 10% of positives edges
        holdout = int(len(positives_subgraph)*0.1)
        random.seed(42)
        blind_test_positives = [(x[0],x[1],x[2]) for x in random.sample(positives_subgraph, holdout)]

        #Subctract the blind_test_positives from the positives subgraph
        positives_subgraph = set(positives_subgraph) - set(blind_test_positives)
        print(f'{len(positives_subgraph)} post blind test')

        #Save the positives for future blind test
        df = pd.DataFrame(blind_test_positives,columns = ["subject","object","predicate"])
        blind_test_positives = [(x[0],x[1],relation.split('-')[1]) for x in blind_test_positives]
        test_dir = 'blind_test_occ'
        save_test_path = f'{test_dir}/{name_view_graph}/{relation}.csv'
        df.to_csv(save_test_path, index=False)
        print(f'Positives for blind test saved in {save_test_path}')
    

    #Apply embedding
    embedding_to_edge = {}
    edges_embs = 0
    X_pos = []
    X_neg = []
    
    if embedding_name == 'n2v' and not load_embedding:
        # apply_node2vec take the path of the folder where you can save the embeddings
        path = f'store_embeddings/{embedding_name}/{name_view_graph}.csv'
        edges_embs = apply_node2vec(subgraph,path) if embedding_subgraph else apply_node2vec(graph,path)
        return 
    elif embedding_name == 'transe' and not load_embedding:
        transe_graph = grape_graph_from_networkx(subgraph,'rna_kg',True) if embedding_subgraph else grape_graph_from_networkx(graph,'rna_kg',False)
        print("Applying TransE embedding...", color='yellow')
        graph_embedding = apply_transe_embedding(transe_graph)
        embedding_nodes = graph_embedding.get_all_node_embedding()

        # Save embeddings to CSV
        os.makedirs('store_embeddings/transe', exist_ok=True)
        df_transe = embedding_nodes[0].copy()
        df_transe.insert(0, 'name', df_transe.index)
        X_transe = df_transe.iloc[:, 1:].to_numpy()
        df_transe['embedding'] = [json.dumps(row.tolist()) for row in X_transe]
        df_transe[['name', 'embedding']].reset_index(drop=True).to_csv(f'store_embeddings/transe/{name_view_graph}.csv', index=False)
        print(f"TransE embeddings saved in store_embeddings/transe/{name_view_graph}.csv", color='green')

    #Load embeddings
    if load_embedding:
        embeddings_path = f'store_embeddings/{embedding_name}/{name_view_graph}.csv'
        df = pd.read_csv(embeddings_path)
        embedding_to_edge =  {
        str(row[0]): np.array(ast.literal_eval(row[1]), dtype=np.float64)
        for _, row in df.iterrows() }
        
        if embedding_name == "n2v":
            all_values = np.concatenate(list(embedding_to_edge.values()))
            global_min, global_max = all_values.min(), all_values.max()

            embedding_to_edge = {
                node_id: 2 * (emb - global_min) / (global_max - global_min) - 1
                for node_id, emb in embedding_to_edge.items()
            }

            all_values = np.concatenate(list(embedding_to_edge.values()))
            print(f'Embedding n2v normalizzati: {np.all(all_values >= -1) and np.all(all_values <= 1)}')     
        
    
        
    #Mapping embeddings
    for p in tqdm(positives_subgraph, desc = "Mapping positives edges embedding"):
        if load_embedding:
            emb = np.multiply(embedding_to_edge[p[0]],embedding_to_edge[p[1]])
        elif embedding_name == 'n2v' and not load_embedding:
            emb = edges_embs[(p[0], p[1])]
        elif embedding_name == 'transe' and not load_embedding:
            df_embedding_nodes = embedding_nodes[0]
            emb_src = df_embedding_nodes.loc[p[0]]
            emb_dst = df_embedding_nodes.loc[p[1]]
            emb = np.multiply(emb_src.tolist(),emb_dst.tolist())

        X_pos.append(emb)
        embedding_to_edge[tuple(emb.tolist())] = p  # Salva la mappatura (embedding → arco)

    for n in tqdm(predicted_negatives, desc = "Mapping predicted negatives edges embedding"):
        if load_embedding:
            emb = np.multiply(embedding_to_edge[n[0]],embedding_to_edge[n[1]])
        elif embedding_name == 'n2v' and not load_embedding:
            emb = edges_embs[(n[0], n[1])]
        elif embedding_name == 'transe' and not load_embedding:
            df_embedding_nodes = embedding_nodes[0]
            emb_src = df_embedding_nodes.loc[n[0]]
            emb_dst = df_embedding_nodes.loc[n[1]]
            emb = np.multiply(emb_src.tolist(),emb_dst.tolist())

        X_neg.append(emb)
        embedding_to_edge[tuple(emb.tolist())] = n  # Salva la mappatura (embedding → arco) 

    # Train and evaluate the model
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    X = np.vstack((X_pos, X_neg))


    print(f'{len(X)} embeddings generated')

    y = np.array([1] * len(X_pos) + [-1] * len(X_neg))

    # Grid
    hidden_layer_values = [(50,), (100,), (100,50), (200,100)]
    alpha_values = np.logspace(-5, -1, 13)   # 1e-5 ... 1e-1
    #---

    print("Training and evaluating models...")
    results_list,tp,fp,tn,fn,tp_arr,fp_arr,tn_arr,fn_arr,best_params = train_and_evaluate(X, y, alpha_values, hidden_layer_values, embedding_to_edge, relation,strategy_name,name_view_graph,embedding_name,parameter_model_selection,dump, combined_negatives = combined_negatives)
    print("Training and evaluation completed")

    pong = time.time()
    tempo_trascorso = int(pong - ping)
    ore = tempo_trascorso // 3600
    minuti = (tempo_trascorso % 3600) // 60
    secondi = tempo_trascorso % 60
    excecution_time = f"{ore:02d}:{minuti:02d}:{secondi:02d}"
    print(f"Total time taken: {excecution_time}\n{name_view_graph} | {relation} | {embedding_name} | {strategy_name} | {parameter_model_selection}", color='m')

    #Calculate mean and dev std of the metrics
    results_mean = calculate_mean_std(results_list)

    #store the results
    results["view_name"] = f'{name_view_graph}'
    results["relation"] = relation
    results["strategy_name"] = strategy_name if not combined_negatives else 'combined_negatives'
    results["embedding"] = embedding_name
    results["nodes_type_src"] = number_type_src
    results["nodes_type_dst"] = numeber_type_dst
    results["total_nodes"] = len(set(list(subgraph.nodes())))
    results["edges"] = len(list(subgraph.edges()))
    results["tp"] = tp
    results["fp"] = fp
    results["tn"] = tn
    results["fn"] = fn
    results["tp1"], results["tp2"], results["tp3"], results["tp4"], results["tp5"] = tp_arr[0], tp_arr[1], tp_arr[2], tp_arr[3], tp_arr[4]
    results["fp1"], results["fp2"], results["fp3"], results["fp4"], results["fp5"] = fp_arr[0], fp_arr[1], fp_arr[2], fp_arr[3], fp_arr[4]
    results["tn1"], results["tn2"], results["tn3"], results["tn4"], results["tn5"] = tn_arr[0], tn_arr[1], tn_arr[2], tn_arr[3], tn_arr[4]
    results["fn1"], results["fn2"], results["fn3"], results["fn4"], results["fn5"] = fn_arr[0], fn_arr[1], fn_arr[2], fn_arr[3], fn_arr[4]
    results["accuracy"] = results_mean["accuracy"]
    results["precision"] = results_mean["precision"]
    results["recall"] = results_mean["recall"]
    results["specificity"] = results_mean["specificity"]
    results["fpr"] = results_mean["fpr"]
    results["fnr"] = results_mean["fnr"]
    results["f_beta_1"] = results_mean["f_beta_1"]
    results["f_beta_2"] = results_mean["f_beta_2"]
    results["error_beta_score_0_5"] = results_mean["error_beta_score_0_5"]
    results["error_beta_score_1"] = results_mean["error_beta_score_1"]
    results["error_beta_score_2"] = results_mean["error_beta_score_2"]
    results["parameter_model_selection"] = parameter_model_selection
    results["excecution_time"] = excecution_time
    results["best_alpha"] = best_params["alpha"]
    results["best_hidden_layer"] = best_params["hidden_layer"]
    results["balanced_accuracy"] = results_mean["balanced_accuracy"]
    results["MCC"] = results_mean["MCC"]


    #NOTE: change this if you want another directory
    if combined_negatives:
        save_dir = f'experiments_new/{name_view_graph}/{embedding_name}/MLPC_combined_negatives_fixed.csv'
    else:
        save_dir = f'experiments_new/{name_view_graph}/{embedding_name}/MLPC_{strategy_name}_fixed.csv'

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(save_dir, mode="a", header=False, index=False,na_rep="NaN")
    
#------------- Schema facts and running experiments -----------------

TYPES_miRNA_KG = [
    'Gene - causes or contributes to condition - Disease',
    # 'Gene - is causal germline mutation in - Disease',
    'miRNA - in similarity relationship with - miRNA',
    # 'miRNA - genetically interacts with - miRNA',
    'miRNA - causes or contributes to condition - Disease',
    'miRNA - under expressed in - Disease',
    'miRNA - over expressed in - Disease',
    'miRNA - is causal somatic mutation in - Disease',
    'Gene - genetically interacts with - Gene',
    'miRNA - regulates activity of - Gene',
    'miRNA - participates in - GO',
    'miRNA - has function - GO',
    'miRNA - located in - GO',
    'miRNA - part of - GO',
    # 'miRNA - under expressed in - GO', 
    # 'miRNA - enables - GO',
    # 'miRNA - acts upstream of - GO'
    ]

# compute(TYPES_miRNA_KG[0],strategy=community_based_negative_sampling,strategy_name='c-b-n-s_1',name_view_graph = 'miRNA-KG',embedding_name = 'transe',parameter_model_selection="error_beta_score_1",load_embedding = True,dump = True,blind_test= True, combined_negatives = False)


TYPES_PKT_KG = [
    'Protein - located in - Anatomy',
    'Protein - located in - Cell',
    # 'Anatomy - participates in - GO', 
    'Protein - molecularly interacts with - Protein', 
    # 'Cell - participates in - GO',
    # 'Cell - capable of - GO',
    # 'Cell - derives from - GO', 
    # 'GO - participates in - GO',
    'GO - negatively regulates - GO', 
    'GO - positively regulates - GO', 
    'GO - regulates - GO',
    # 'GO - directly positively regulates - GO',
    # 'GO - directly negatively regulates - GO', 
    # 'GO - directly regulates - GO', 
    'Chemical - participates in - GO',
    'Chemical - molecularly interacts with - GO',
    # 'Chemical - located in - GO',
    # 'Protein - participates in - GO', 
    'Protein - enables - GO',
    'Protein - located in - GO',
    # 'Protein - contributes to - GO', 
    'Chemical - interacts with - Gene',
    # 'Chemical - molecularly interacts with - Gene', 
    'Chemical - interacts with - Protein',
    'Chemical - molecularly interacts with - Protein',
    'Chemical - is substance that treats - Disease',
    'Chemical - participates in - Pathway',
    'Protein - participates in - Pathway',
    'Gene - causes or contributes to condition - Disease',
    # 'Gene - is causal germline mutation in - Disease',
    'Gene - genetically interacts with - Gene',
    'Gene - interacts with - Protein',
    'Gene - participates in - Pathway'
]

# compute(TYPES_PKT_KG[0],strategy=community_based_negative_sampling,strategy_name='c-b-n-s_1',name_view_graph = 'PKT-KG',embedding_name = 'distmult',parameter_model_selection= "error_beta_score_1",dump= True,load_embedding=True,blind_test=True, combined_negatives = False)


TYPES_Hetionet = [
    'Anatomy - downregulates - Gene',
    'Anatomy - expresses - Gene',
    'Anatomy - upregulates - Gene',
    'Compound - binds - Gene',
    'Compound - downregulates - Gene',
    'Compound - upregulates - Gene',
    'Compound - causes - Side_effect',
    # 'Compound - palliates - Disease',
    # 'Compound - treats - Disease', 
    'Compound - resembles - Compound',
    'Disease - associates - Gene',
    'Disease - downregulates - Gene',
    'Disease - upregulates - Gene',
    'Disease - localizes - Anatomy',
    'Disease - presents - Symptom', 
    # 'Disease - resembles - Disease',
    'Gene - covaries - Gene',
    'Gene - interacts - Gene',
    'Gene - regulates - Gene', 
    'Gene - participates - Biological_process', 
    'Gene - participates - Cellular_component', 
    'Gene - participates - Molecular_function', 
    'Gene - participates - Pathway',
    'Pharmacologic_class - includes - Compound'
]

# compute(TYPES_Hetionet[0],strategy=community_based_negative_sampling,strategy_name='c-b-n-s',name_view_graph = 'Hetionet',embedding_name = 'transe',parameter_model_selection= "error_beta_score_1",dump= True,load_embedding=True,blind_test=True, combined_negatives = False)

                
TYPES_PrimeKG = [
    'Anatomy - expression absent - Gene_and_or_protein',
    'Anatomy - expression present - Gene_and_or_protein',
    'Biological_process - interacts with - Exposure', 
    'Biological_process - interacts with - Gene_and_or_protein',
    'Cellular_component - interacts with - Gene_and_or_protein',
    'Disease - associated with - Gene_and_or_protein',
    'Disease - contraindication - Drug',
    'Disease - indication - Drug',
    'Disease - off label use - Drug', #9 off-label
    'Disease - linked to - Exposure',
    'Disease - phenotype absent - Effect_and_or_phenotype',
    'Disease - phenotype present - Effect_and_or_phenotype', 
    # 'Drug - carrier - Gene_and_or_protein',
    'Drug - enzyme - Gene_and_or_protein',
    'Drug - target - Gene_and_or_protein',
    'Drug - transporter - Gene_and_or_protein',
    'Drug - side effect - Effect_and_or_phenotype',
    'Drug - synergistic interaction - Drug', 
    'Effect_and_or_phenotype - associated with - Gene_and_or_protein',
    'Exposure - interacts with - Gene_and_or_protein', 
    'Gene_and_or_protein - interacts with - Molecular_function',
    'Gene_and_or_protein - interacts with - Pathway',
    'Gene_and_or_protein - ppi - Gene_and_or_protein'
]

# compute(TYPES_PrimeKG[0],strategy=community_based_negative_sampling,strategy_name='c-b-n-s',name_view_graph = 'PrimeKG',embedding_name = 'transe',parameter_model_selection= "error_beta_score_1",dump= True,load_embedding=True,blind_test=True, combined_negatives = False)

TYPES_OptimusKG = [
    'Anatomy - EXPRESSION_ABSENT - Gene',               
    'Anatomy - EXPRESSION_PRESENT - Gene',              
    'Drug - ASSOCIATED_WITH - Phenotype',               
    'Drug - CONTRAINDICATION - Phenotype',              
    'Drug - INDICATION - Phenotype',                    
    'Drug - OFF_LABEL_USE - Disease',                   
    'Drug - CONTRAINDICATION - Disease',                
    'Drug - INDICATION - Disease',                      
    'Drug - ENZYME - Gene',                             
    'Drug - TARGET - Gene',                             
    'Drug - TRANSPORTER - Gene',                        
    'Biological_process - INTERACTS_WITH - Gene',       
    'Cellular_component - INTERACTS_WITH - Gene',       
    'Disease - ASSOCIATED_WITH - Gene',                 
    'Pathway - INTERACTS_WITH - Gene',                  
    'Disease - PHENOTYPE_PRESENT - Phenotype',          
    'Drug - SYNERGISTIC_INTERACTION - Drug',            
    'Exposure - INTERACTS_WITH - Biological_process',   
    'Exposure - INTERACTS_WITH - Gene',                 
    'Exposure - LINKED_TO - Disease',                   
    'Phenotype - ASSOCIATED_WITH - Gene',               
    'Gene - INTERACTS_WITH - Gene',                     
    'Molecular_function - INTERACTS_WITH - Gene'        
]

# compute(TYPES_OptimusKG[0],strategy=community_based_negative_sampling,strategy_name='c-b-n-s',name_view_graph = 'OptimusKG',embedding_name = 'transe',parameter_model_selection= "error_beta_score_1",dump= True,load_embedding=True,blind_test=True, combined_negatives = False)