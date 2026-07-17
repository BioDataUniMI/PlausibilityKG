# Plausibility Scoring for candidate biomedical edges

Research codebase for scoring the **plausibility of predicted edges** in biomedical knowledge graphs (PKT-KG, miRNA-KG, Hetionet, PrimeKG, OptimusKG, and others). Given a knowledge graph and a target schema fact, the pipeline generates negative edges, embeds the graph, trains a classifier to separate real from negative edges and evaluates how plausible new (candidate) edges are.

## Pipeline overview

For a given schema fact (e.g. `Gene - interacts - Gene`) and knowledge graph view:

1. **Load & filter**
   - Read `data/edges_<view>.csv` and `data/nodes_<view>.csv`, and join node types onto edges.
   - Build a single graph from **all** edges/nodes of the KG view (not just the target schema fact) — this is the graph that gets embedded in step 3.
   - Separately, filter that data down to a **subgraph** containing only edges of the target schema fact (e.g. `Gene - interacts - Gene`). This subgraph is only used to pick which node pairs become positive training examples — it is not what gets embedded.
2. **Negative sampling** — generate negative edges using one of the strategies in [`plausibility/strategies.py`](plausibility/strategies.py) (e.g. random, degree-aware, community-based, shortest-path-based, PageRank-based), or load a previously computed set from `negative_samples/<view>/<strategy_name>/<schema_fact>.csv`.
3. **Embedding**
   - Compute or load node embeddings (TransE via `grape`) for the **full KG graph** built in step 1 — the whole KG view is embedded, regardless of which schema fact is being scored.
   - The positive edges (from the schema fact's subgraph) and negative edges (from step 2) then each look up their two endpoint embeddings from that embedding and combine them (element-wise product) into a single feature vector per edge — this is the input to the classifier in step 4.
4. **Classification** — train a classifier (Random Forest or MLP) to distinguish positive from negative edges, with hyperparameter selection via cross-validation.
5. **Evaluation** — compute accuracy, precision, recall, specificity, F-beta, and error-beta scores; optionally hold out a blind test set of positive edges.
6. **Plausibility score** — use the trained model to score a candidate edge. Core logic lives in `plausibility_score_computation.py`; to score a specific triple, use `plausibility_score_computation.ipynb`, which calls `compute_plausibility_score(schema_fact, source_id, target_id)` with worked examples for each KG.

## Data & Models

The KGs (nodes and edges) used in this project are available [here]().

The trained models (Random Forest and MLP) are available [here]().

## Project structure

```
plausibility/                     Core library
├── dataset.py                    Dataset download/read helpers
├── strategies.py, strategies_new.py   Negative sampling strategies
├── samplings.py                  Sampling utilities
├── embedding_utils.py            Embedding helpers (TransE / node2vec)
├── classifiers.py                Custom classifiers (e.g. OCSVM wrapper)
├── metrics.py                    Evaluation metrics
└── visualizations.py             Plotting helpers

plausibility_computation_realKG_RF.py     Random Forest classifier — defines a
                                           `compute(...)` entry point
plausibility_computation_realKG_MLPC.py   MLP classifier — defines a
                                           `compute(...)` entry point

plausibility_score_computation.py      Plausibility scoring logic
plausibility_score_computation.ipynb   Entry point to score a single triple
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

The classifier scripts don't take CLI arguments — each script defines its own `compute(...)` function and a list of schema facts at the bottom of the file. To run an experiment:

1. Open the script for the classifier you want, e.g. [`plausibility_computation_realKG_RF.py`](plausibility_computation_realKG_RF.py).
2. Uncomment / edit a call to `compute(...)` at the bottom of the file, e.g.:
   ```python
   compute(
       TYPES_Hetionet[0],
       strategy=community_based_negative_sampling,
       strategy_name='c-b-n-s',
       name_view_graph='Hetionet',
       embedding_name='transe',
       parameter_model_selection='error_beta_score_1',
       dump=True,
       load_embedding=True,
       blind_test=True,
       combined_negatives=False,
   )
   ```
3. Run it:
   ```bash
   python plausibility_computation_realKG_RF.py
   ```

Key `compute()` parameters:

| Parameter | Meaning |
|---|---|
| `relation` | Schema fact used to build the subgraph (view-specific, see `TYPES_*` lists) |
| `strategy` / `strategy_name` | Negative sampling function + short name used for caching |
| `name_view_graph` | KG view to use (`PKT-KG`, `miRNA-KG`, `Hetionet`, `PrimeKG`, `OptimusKG`) |
| `embedding_name` | `'transe'` |
| `load_embedding` | Reuse a cached embedding instead of recomputing |
| `dump` | Save the trained model |
| `blind_test` | Hold out 10% of positive edges for a later blind evaluation |
| `combined_negatives` | Use a precombined negative set instead of a single strategy |

Results (metrics, trained models) are written under `experiments_new/` and `dumps_models_new/`.