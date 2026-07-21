\# dumps\_models/



This folder holds the trained plausibility classifiers for each bioKG used in this project.



\## Setup



1\. Download the trained models archive from \[Zenodo]().

2\. Extract it.

3\. Copy the model files into this folder, following this structure per bioKG:



dumps\_models/

├── <kg>/

│     └── transe/

│           └── c-b-n-s/

│                 └── err\_beta\_1/

│                       └── RF/

│                             ├── <schema\_fact\_1>.pkl

│                             ├── <schema\_fact\_2>.pkl

│                             ...

├── <kg\_2>/

│     └── transe/c-b-n-s/err\_beta\_1/RF/...

...





Each bioKG has its own top-level folder named after it (e.g. `dumps\_models/miRNA-KG/`), and every path below that must match exactly — `transe/c-b-n-s/err\_beta\_1/RF/` — since the code reads models from this exact path.



See the main \[README](../README.md#data-and-pretrained-models) for the list of supported bioKGs.



