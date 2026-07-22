# dumps_models/

This folder holds the trained plausibility classifiers for each bioKG used in this project.

## Setup

1. On [Zenodo](https://doi.org/10.5281/zenodo.21359878), download the zip file for each bioKG you need (each bioKG has its own separate zip).
2. Extract a zip — it gives you one folder named after that bioKG (e.g. `miRNA-KG/`).
3. Inside it, there is a `dumps_models/` subfolder. Copy that subfolder here, renaming it to the bioKG's name — e.g. copy `miRNA-KG/dumps_models/` and place it here as `dumps_models/miRNA-KG/`.

Each model file's full path must end up looking exactly like this:

`dumps_models/<kg>/transe/c-b-n-s/err_beta_1/RF/<schema_fact>.pkl`

For example: `dumps_models/miRNA-KG/transe/c-b-n-s/err_beta_1/RF/miRNA - participates in - GO.pkl`

Repeat for each bioKG's zip — every bioKG gets its own top-level folder named after it, with the same `transe/c-b-n-s/err_beta_1/RF/` path underneath, since the code reads models from this exact path.
