# SEC EEG pipeline

Process EDF recordings and scored hypnograms into FIF files, sleep features, and EEG summary reports using MNE, YASA, and FOOOF.

## Install

Use **Python 3.12**. On macOS, install it with `brew install python@3.12` if needed.

From the repository root:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name sec-eeg --display-name "SEC EEG"
```

On macOS, a missing `libomp.dylib` error can be resolved with `brew install libomp`.

## Setup and run

Set your input and output paths in `config.toml`.

By default, place EDF files in `data/edf/` and matching hypnogram CSVs in `data/hypnogram/`. 

```bash
bash run_pipeline.sh
```

The pipeline exports FIF files to `data/fif/`, feature CSVs to `data/features/`, and notebook/HTML reports to `report/eeg_summary/`. Output folders are created automatically; existing outputs are skipped.
