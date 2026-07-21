# dumps_models/

This folder holds the trained plausibility classifiers for each bioKG used in this project.

## Setup

1. Download the trained models archive from [Zenodo]().
2. Extract it. Inside, you'll find one folder per bioKG (e.g. `miRNA-KG/`).
3. Inside each bioKG's folder, there is a `dumps_models/` subfolder. Copy that subfolder here, renaming it to the bioKG's name — e.g. copy `miRNA-KG/dumps_models/` and place it here as `dumps_models/miRNA-KG/`.

Each model file's full path must end up looking exactly like this:

`dumps_models/<kg>/transe/c-b-n-s/err_beta_1/RF/<schema_fact>.pkl`

For example: `dumps_models/miRNA-KG/transe/c-b-n-s/err_beta_1/RF/miRNA - participates in - GO.pkl`

Repeat for every bioKG — each one gets its own top-level folder named after it, with the same `transe/c-b-n-s/err_beta_1/RF/` path underneath, since the code reads models from this exact path.
