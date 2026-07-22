# data/

This folder holds the node and edge files for each bioKG used in this project.

## Setup

1. On [Zenodo](https://doi.org/10.5281/zenodo.21359878), download the zip file for each bioKG you need (each bioKG has its own separate zip).
2. Extract a zip — it gives you one folder named after that bioKG (e.g. `miRNA-KG/`).
3. Inside it, there is a `data/` subfolder containing two files: `<kg>_nodes.csv` and `<kg>_edges.csv`. Copy both here, **renaming them** so the bioKG name moves from a prefix to a suffix:

   - `miRNA-KG/data/miRNA-KG_nodes.csv` → `data/nodes_miRNA-KG.csv`
   - `miRNA-KG/data/miRNA-KG_edges.csv` → `data/edges_miRNA-KG.csv`

Repeat for each bioKG's zip — you should end up with one `nodes_<kg>.csv` and one `edges_<kg>.csv` pair per bioKG, all sitting directly in this folder.

