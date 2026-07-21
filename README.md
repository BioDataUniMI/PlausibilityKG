# Plausibility-Driven Prioritization of Candidate Biomedical Annotations

This repository provides the implementation of a plausibility-based approach for supporting biomedical annotation and curation.

Given a candidate biomedical fact, the pipeline estimates how compatible it is with the patterns already encoded in a bioKG. The resulting plausibility scores can help curators prioritize strongly supported candidates, identify likely implausible annotations, and flag uncertain cases that require expert inspection.

## Pipeline overview

Given a bioKG, such as Hetionet or PrimeKG, and one of its schema facts, such as `Gene - interacts - Gene`, the pipeline performs the following steps.

1. **Load and filter the bioKG**
   - Read `data/edges_<kg>.csv` and `data/nodes_<kg>.csv`.
   - Associate source and target node types with each edge.
   - Select the edges associated with the target schema fact. This partition determines the positive examples for the corresponding classifier.

2. **Generate negative examples**
   - Generate negative edges using one of the strategies implemented in [`plausibility/strategies.py`](plausibility/strategies.py), including the community-based sampling presented in the article.
   - Alternatively, load a previously generated negative set from:

     ```text
     negative_samples/<kg>/<strategy_name>/<schema_fact>.csv
     ```

3. **Compute graph embeddings**
   - Compute or load node embeddings for the complete bioKG using [`grape`](https://github.com/AnacletoLAB/grape).
   - Supported embedding methods include TransE, ComplEx, TransH, DistMult, RotatE, and Node2Vec.
   - For each positive or negative edge, retrieve the embeddings of its source and target nodes.
   - Combine the two endpoint embeddings through an element-wise Hadamard product.
   - The resulting edge-level feature vector is used as input to the binary classifier.

4. **Train a relation-specific classifier**
   - Train a binary classifier for the selected schema fact to distinguish observed positive edges from generated negative examples. Random Forest and Multilayer Perceptron classifiers are supported.
   - Hyperparameters are selected through cross-validation.
   - The classifier output estimates the membership of a candidate edge to the target schema fact, i.e., its compatibility with the structural and semantic patterns observed for that relation type in the bioKG.

5. **Evaluate the classifier**
   - Compute classification metrics including:
     - balanced accuracy;
     - precision;
     - recall;
     - specificity;
     - F-beta score;
     - error-beta score (as described in the article).
   - Optionally, hold out a blind set containing 10% of the observed positive edges. Blind positive edges are excluded from training and model selection and are subsequently used to evaluate plausibility on unseen positive facts (it is useful for replicating experiments in the article).

6. **Compute plausibility scores**
   - Use the trained relation-specific classifiers to estimate the plausibility of candidate biomedical annotations. Estimates can be computed according to the Base, Gain, SoftMax, and COmbo formulations presented in the article.
   - See:
     - [`plausibility_score_computation.py`](plausibility_score_computation.py)
     - [`plausibility_score_computation.ipynb`](plausibility_score_computation.ipynb)

   - You can see the methods implemented for evaluating the quality of plausibility scores in:
     - [`plausibility_score_metrics.py`](plausibility_score_metrics.py)

## Data and pretrained models

The bioKG datasets, embeddings, generated negative samples, and pretrained models used in the experiments are available [here]().

The currently supported bioKGs are:

- [PKT-KG](https://doi.org/10.1038/s41597-024-03171-w)
- [miRNA-KG](https://doi.org/10.1093/nargab/lqaf194)
- [Hetionet](https://doi.org/10.7554/eLife.26726)
- [PrimeKG](https://doi.org/10.1038/s41597-023-01960-3)
- [OptimusKG](https://doi.org/10.48550/arXiv.2604.27269)

## Project structure

```text
plausibility/
├── dataset.py                         Dataset download and loading utilities
├── strategies.py                      Negative-sampling strategies
├── samplings.py                       General sampling utilities
├── embedding_utils.py                 Graph-embedding utilities
└── metrics.py                         Classification and evaluation metrics


[MIAD: let's add the empty subfolders here with readme pointing to zenodo and a minimal instruction on what to download and place inside]


plausibility_computation_realKG_RF.py   Random Forest experiments
plausibility_computation_realKG_MLPC.py MLP experiments

plausibility_score_computation.py       Plausibility-score computation
plausibility_score_computation.ipynb    Interactive notebook for scoring
                                        candidate biomedical annotations
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

plausibility_computation_realKG_{RF/MLPC}.py scripts define a `compute(...)` function and the schema-facts we considered in the article.

To run an experiment:

1. Open the script corresponding to the desired classifier, for example:

   [`plausibility_computation_realKG_RF.py`](plausibility_computation_realKG_RF.py)

2. Add, uncomment, or modify a call to `compute(...)`:

   ```python
   compute(
       relation="Disease - associates - Gene",
       strategy=community_based_negative_sampling,
       strategy_name="c-b-n-s",
       name_view_graph="Hetionet",
       embedding_name="transe",
       parameter_model_selection="error_beta_score_1",
       dump=True,
       load_embedding=True,
       blind_test=True,
   )
   ```

3. Run the script:

   ```bash
   python plausibility_computation_realKG_RF.py
   ```

## Main `compute()` parameters

| Parameter | Description |
|---|---|
| `relation` | Target schema fact, for example `Disease - associates - Gene` |
| `strategy` | Negative-sampling function |
| `strategy_name` | Short identifier used to cache and retrieve generated negatives |
| `name_view_graph` | BioKG to use: `PKT-KG`, `miRNA-KG`, `Hetionet`, `PrimeKG`, or `OptimusKG` |
| `embedding_name` | Embedding method: `transe`, `node2vec`, `complex`, `transh`, `distmult`, or `rotate` |
| `parameter_model_selection` | Metric used during classifier model selection |
| `load_embedding` | Reuse a cached embedding instead of recomputing it |
| `dump` | Save the trained classifier |
| `blind_test` | Hold out 10% of positive edges for the blind plausibility evaluation used in the article |

Experimental metrics and trained models are written to:

```text
experiments/
dumps_models/
```

## Scoring a candidate biomedical annotation

[`plausibility_score_computation.ipynb`](plausibility_score_computation.ipynb) assigns a plausibility score to a candidate annotation.

The notebook loads the required bioKG embeddings and trained relation-specific classifiers and computes the plausibility scores associated with a candidate biomedical fact.

A candidate annotation is represented as a triple:

```text
subject - predicate - object
```

For example:

```text
Disease - associates - Gene
```

The candidate entities must use identifiers compatible with those adopted in the selected bioKG.

## Plausibility formulations

The repository implements four plausibility formulations:

- **Base**: uses the confidence assigned by the classifier associated with the target schema fact.
- **Gain**: compares the target classifier score with the strongest competing predicate.
- **SoftMax**: normalizes the target score against the most relevant competing predicates.
- **Combo**: combines the local classifier confidence with the competition-aware gain.

The competition-aware formulations are designed for cases in which multiple biologically meaningful predicates may connect the same pair of entities. Details are provided in the article.

The approach is designed as a support mechanism for assisted biomedical annotation and curation. The final decision remains under the control of expert biomedical curators.

## Citation

Please cite the following articles if this project was useful for your research:

```bibtex
  @article{Cavalleri2026plausibilitykg,
      title="Plausibility-Driven Prioritization of Candidate Biomedical Annotations", 
      author="Emanuele Cavalleri and Miad Alavinezhad and Dario Malchiodi and Marco Mesiti",
      ADD ARXIV WHEN PUBLISHED
  }
```
