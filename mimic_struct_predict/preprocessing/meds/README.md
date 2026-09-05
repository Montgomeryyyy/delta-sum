## MEDS Preprocessing

> TL;DR The preprocessing has been updated from the MIMIC_IV_MEDS example to allow for natural language rep and newer version of med-transforms to be used

The idea of this preprocessing pipeline is to use the ideas/best practices from MEDS to produce a MEDS formatted dataset which can be used for tabular, structured and natural language modelling downstream.

The basis for this was the [MIMIC_IV_MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS). However - It uses an old version of [meds-transforms](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS/blob/4f4b6a1dc2aaf318038726863b400632beb2b4ff/pyproject.toml#L21) which was only compatible with a very early version of meds-tab (v0.0.2).
- To make the natural language format work many joins to mimic dictionaries was required and so an upgrade of the pre_MEDS.py script
- Updating of the event_config was inevitable to allow for the use of natural language descriptions for events

So this module instead uses the med-transform pipeline example from [meds-extract](https://github.com/mmcdermott/MEDS_extract/blob/51eef880722722ec9289b15358f5f33f70ea53e8/src/MEDS_extract/configs/_extract.yaml)


## Usage

`make meds-dataset`
1. Runs pre_MEDS.py
2. Runs `ETL.yaml` pipeline which points at `event_configs.yaml`

## Directory Structure

- `pre_MEDS.py`: Runs steps that cannot be easily configured in MEDS
- `config/`: Configuration files for the pipeline.
    - `ETL.yaml`: Defines stages of the preprocessing pipeline.
    - `event_configs.yaml`: Defines the mapping of MIMIC tables -> MEDS format
- `utils/`: Utility functions for the pipeline (used in MIMIC_IV_MEDS pre_MEDS.py but removed from newer versions of MEDS_transforms so copied and pasted in)
    - `extract_utils.py`: Utility functions for extracting data from MIMIC-IV.
    - `utils.py`: General utility functions.
