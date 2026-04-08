# mimic-struct-predict


## Implemented

### Data prep

Download mimic
- `make download-mimic`
- Can comment out for demo or full dataset
- Requires physionet credentials
- Output is in `data/raw_input`

Convert to MEDS
- `make meds-dataset`
- Multiple stage output. Notably:
    - tabular data
        - `data/MEDS_cohort/add_age_rows`
    - coded data
        - `data/MEDS_cohort/bin_numeric_values`
        - Has age as string rather than numeric and quantiled numeric values
    - natural language data
        - `data/MEDS_cohort/data`
        - Has numeric values as strings appended to codes

Create ACES labels
- `make aces-labels`
- Output is in `data/aces_labels`



uv run meds-tab-describe "MEDS_cohort_dir=data/MEDS_cohort" "input_dir=data/MEDS_cohort/add_age_rows/train"


uv run meds-tab-cache-task \
    "MEDS_cohort_dir=data/MEDS_cohort" \
    "input_dir=data/MEDS_cohort/add_age_rows" \
    "output_dir=data/MEDS_cohort/30_day_readmission/task_cache" \
    "input_label_dir=data/aces_labels/30_day_readmission" \
    "task_name=30_day_readmission" \
    "tabularization.min_code_inclusion_frequency=10" \
    "tabularization.window_sizes=[1d,30d,365d,full]" \
    "tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"




## Plan

### Data prep

- ~~Download meds~~
- ~~premeds~~
- ~~meds pipeline~~
- ~~aces~~
    - Add more tasks

### Models

#### Tabular (XBoost/Meds-tab)

- Train
- Inference

#### Coded (Huggingface)

- Train
    - Supervised ft
    - Embeddings
- Inference
    - Generative rollout
    - Embeddings

#### Natural Language

- Train
    - Instruction tune
- Inference
    - Zero shot
        - Local
        - Frontier
    - Instruction tuned

## Evaluation

- AUROC
- AUPRC
- Brier Score
- Throughput


## Nice to haves

- Better handling of demo vs full dataset in Makefile
- GRPO training
