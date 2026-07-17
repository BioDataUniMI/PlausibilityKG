from typing import Callable
import csv
import pathlib
import requests

import networkx as nx


def download_dataset(url : str, verbose : bool = False) -> pathlib.Path:
    """
    Downloads a dataset from a given URL and saves it to the 'data' directory.

    Parameters
    ----------
    url : str
        The URL of the dataset to download.
    verbose : bool, optional
        If True, prints a message if the file already exists. Defaults to False.

    Returns
    -------
    pathlib.Path
        The path to the downloaded dataset file.

    Raises
    ------
    requests.exceptions.RequestException
        If there is an issue with the network request.
    IOError
        If there is an issue writing the file to disk.
    """
    file_name = f"data/{url.split('/')[-1]}"
    p = pathlib.Path(file_name)
    if p.is_file() and verbose:
            print(f'{file_name} already existing, skipping.')
    else:
        with requests.Session() as s:
            content = s.get(url).content.decode('utf-8-sig')
            cr = csv.reader(content.splitlines(), delimiter=',')
            rows = list(cr)
            with open(p, 'w', newline='') as f:
                cv = csv.writer(f)
                cv.writerows(rows)
    return p

def read_dataset(file_name : str, pos_label_criterion : Callable[[str], bool],real_kg: bool):
    """
    Reads a dataset from a CSV file and creates a directed graph.

    Parameters
    ----------
    file_name : str
        The path to the CSV file containing the dataset.
    pos_label_criterion : function
        A function that takes a label as input and returns True if the edge is positive, False otherwise.

    Returns
    -------
    g : networkx.DiGraph
        A directed graph created from the dataset.
    pos_edges : set of tuples
        A set of positive edges in the form (source, dest, label).
    neg_edges : set of tuples
        A set of negative edges in the form (source, dest, label).

    Raises
    ------
    AssertionError
        If there are edges that are both in positive and negative sets.
    """
    g = nx.DiGraph()

    pos_edges = []
    neg_edges = []
    
    with open(file_name, 'r') as f:
        cr = csv.reader(f)
        for source, dest, label in cr:
            g.add_edges_from([(source, dest,{'label' : label})])
            if not real_kg:
                if pos_label_criterion(label):
                    pos_edges.append((source, dest, label))
                else:
                    neg_edges.append((source, dest, label))

    neg_edges = set(neg_edges)
    pos_edges = set(pos_edges)
    assert not neg_edges & pos_edges
    return g, pos_edges, neg_edges