from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import networkx as nx
import numpy as np
import os
from tabulate import tabulate

def visualize_negatives(true_negatives, predicted_negatives, save_path=None):
    """
    Visualizza il confronto tra negativi effettivi e predetti
    usando un diagramma di Venn.
    """
    
    # Converti i negativi effettivi in formato (node1, node2) per confronto
    true_neg_pairs = {(e[0], e[1]) for e in true_negatives}
    print('visualize_negatives')
    print(f'True neg pairs: {true_neg_pairs}')
    pred_neg_pairs = set(predicted_negatives)
    print(f'Pred neg pairs{pred_neg_pairs}')

    # Calcola intersezione e differenze
    correct_predictions = true_neg_pairs.intersection(pred_neg_pairs)
    # print(correct_predictions)
    
    # Crea figura
    plt.figure(figsize=(10, 8))
    
    # Crea diagramma di Venn
    v = venn2([true_neg_pairs, pred_neg_pairs], 
              set_labels=('Negativi Effettivi', 'Negativi Predetti'))
    
    # Aggiungi titolo e statistiche
    plt.title('Confronto tra Negativi Effettivi e Predetti', fontsize=14)
    
    # Calcola percentuali
    precision = len(correct_predictions) / len(pred_neg_pairs) * 100 if pred_neg_pairs else 0
    recall = len(correct_predictions) / len(true_neg_pairs) * 100 if true_neg_pairs else 0
    
    # Aggiungi testo con statistiche
    stats_text = (
        f"Negativi effettivi: {len(true_neg_pairs)}\n"
        f"Negativi predetti: {len(pred_neg_pairs)}\n"
        f"Intersezione: {len(correct_predictions)}\n"
        f"Precisione: {precision:.2f}%\n"
        f"Recall: {recall:.2f}%"
    )
    
    plt.figtext(0.7, 0.2, stats_text, bbox=dict(facecolor='white', alpha=0.7))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='png')
        # print(f"Grafico salvato in {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Restituisci statistiche
    return {
        'true_negatives': len(true_neg_pairs),
        'predicted_negatives': len(pred_neg_pairs),
        'intersection': len(correct_predictions),
        'precision': precision / 100,
        'recall': recall / 100
    }

def draw_fold_graph(graph, correct_edges, fold, save_drawings_path='', verbose=False):
    """
    Draws the graph for a specific fold with edges colored and styled based on correctness and labels.
    
    Parameters:
    - graph: The graph object (networkx.DiGraph).
    - correct_edges: Dictionary mapping edges to a tuple (is_correct, label).
    - fold: The fold number (int).
    """
    edges = list(correct_edges.keys())
    layout = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 8))
    plt.title(f"Fold {fold} graph", fontsize=14)
    nx.draw(graph, pos=layout, with_labels=False, node_size=60, arrows=False, edgelist=[])
    edge_colors = ['g' if (correct_edges[edge][1] == 1) else 'r' for edge in edges]
    edge_styles = ['solid' if correct_edges[edge][0] else 'dashed' for edge in edges]

    legend_elements = [
        Line2D([0], [0], color='g', lw=2, label='Edge positivo predetto correttamente'),
        Line2D([0], [0], color='g', lw=2, linestyle='dashed', label='Edge positivo predetto erroneamente'),
        Line2D([0], [0], color='r', lw=2, label='Edge negativo predetto correttamente'),
        Line2D([0], [0], color='r', lw=2, linestyle='dashed', label='Edge negativo predetto erroneamente')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    nx.draw_networkx_edges(graph, pos=layout, edgelist=edges, edge_color=edge_colors, alpha=.5, width=2, arrows=True, style=edge_styles)
    
    if save_drawings_path:
        plt.savefig(os.path.join(save_drawings_path, f"fold_{fold}.png"), bbox_inches='tight', format='png')
        if verbose:
            print(f"Fold {fold} graph saved at {save_drawings_path}")

    plt.show()

def print_results_table(results):
    """
    Prints the evaluation results in a well-formatted table.
    """
    headers = ["Fold", "Best Nu", "Best Gamma", "Accuracy", "Precision", "Recall", "Specificity", "F-beta"]

    # Extract data for table
    table_data = [[
        res["fold"], res["best_nu"], res["best_gamma"], res["accuracy"],
        res["precision"], res["recall"], res["specificity"], res["f_beta"]
    ] for res in results]

    # Compute averages
    avg_values = {
        "accuracy": np.mean([res["accuracy"] for res in results]),
        "precision": np.mean([res["precision"] for res in results]),
        "recall": np.mean([res["recall"] for res in results]),
        "specificity": np.mean([res["specificity"] for res in results]),
        "f_beta": np.mean([res["f_beta"] for res in results])
    }

    # Add average row
    avg_row = ["AVG", "", "", round(avg_values["accuracy"], 3), round(avg_values["precision"], 3),
               round(avg_values["recall"], 3), round(avg_values["specificity"], 3), round(avg_values["f_beta"], 3)]
    table_data.append(avg_row)

    # Print the table
    print("\n" + tabulate(table_data, headers=headers, tablefmt="pretty") + "\n")

def draw_graph_strategy(graph, title, positives=None, negatives=None, predicted_negatives=None, unknowns=None, save_drawings_path=''):
    plt.figure(figsize=(10, 8))
    plt.title(title)
    layout = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos=layout, with_labels=False, node_size=60, arrows=False, edgelist=[])

    if positives and not negatives:
        for pos in positives:
            nx.draw_networkx_edges(graph, pos=layout, edgelist=[pos], edge_color='green', alpha=.5, width=2, arrows=True, style='solid')
        if save_drawings_path != '':
            plt.savefig(os.path.join(save_drawings_path, "positive_strategy_graph.png"), bbox_inches='tight', format='png')

    if negatives and predicted_negatives:
        for neg in predicted_negatives:
            if neg in negatives:
                nx.draw_networkx_edges(graph, pos=layout, edgelist=[neg], edge_color='red', alpha=.5, width=2, arrows=True, style='solid')
            elif neg in positives:
                nx.draw_networkx_edges(graph, pos=layout, edgelist=[neg], edge_color='red', alpha=.5, width=2, arrows=True, style='dashed')
        if save_drawings_path != '':
            plt.savefig(os.path.join(save_drawings_path, "negative_strategy_graph.png"), bbox_inches='tight', format='png')

    if unknowns:
        for x in unknowns:
            nx.draw_networkx_edges(graph, pos=layout, edgelist=[x], edge_color='#9900CC80', alpha=.5, width=2, arrows=True, style='solid')
        if save_drawings_path != '':
            plt.savefig(os.path.join(save_drawings_path, "unknown_strategy_graph.png"), bbox_inches='tight', format='png')
    plt.show()