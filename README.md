# mimic-struct-predict

## CREATE Setup commands

Get an interative session
`srun -p cpu --time 60 --pty /bin/bash -l`

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

```
export MIMIC_USER=
export MIMIC_PASSWORD=
export SCRATCH_DIR=/scratch$(pwd)
source scripts/config.sh
```
Run in interactive shell
```
uv run hf auth login
bash scripts/download_hf_model.sh
```

```
sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks=96 \
    --mem=512G \
    --time 30 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/setup_uv/%j.out \
    scripts/setup_uv.sh


sbatch\
    --partition cpu \
    --time 600 \
    --mem 2G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,MIMIC_USER,MIMIC_PASSWORD, \
    --output=/scratch/users/%u/logs/download_mimic/%j.out \
    scripts/download_mimic.sh

sbatch\
    --partition cpu \
    --time 600 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=256G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/meds-dataset/%j.out \
    scripts/meds-dataset.sh

sbatch\
    --partition cpu \
    --time 10 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=32G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/meds-dataset/%j.out \
    scripts/meds-dataset.sh

sbatch\
    --partition cpu \
    --time 600 \
    --nodes=1 \
    --cpus-per-task=124 \
    --mem=950G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/meds-tab/%j.out \
    scripts/meds-tab.sh

sbatch\
    --partition cpu \
    --cpus-per-task=8 \
    --mem=256G \
    --time=15 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/natural-language-dataset-prompts/%j.out \
    scripts/natural-language-dataset-prompts.sh

sbatch\
    --partition cpu \
    --cpus-per-task=4 \
    --mem=32G \
    --time=15 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/natural-language-dataset-prompts/%j.out \
    scripts/natural-language-dataset-prompts.sh

sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --mem=16G \
    --time 40 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/natural_language_inference/%j.out \
    scripts/natural_language_inference.sh \
    --config-name inference_thinking_yes_no.yaml \
    --dataset-name thinking_yes_no \
    --split tuning

bash scripts/natural_language_inference.sh \
    --config-name inference_direct.yaml \
    --dataset-name direct_yes_no \
    --split tuning

sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --mem=16G \
    --time 10 \
    --chdir $(pwd) \
    --get-user-env \
    --output=/scratch/users/%u/logs/natural_language_inference/%j.out \
    scripts/natural_language_inference.sh \
    --config-name inference_thinking_probability.yaml \
    --dataset-name thinking_probability \
    --split tuning

sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --mem=16G \
    --time 10 \
    --chdir $(pwd) \
    --get-user-env \
    --output=/scratch/users/%u/logs/vllm_test/%j.out \
    outputs/vllm_test.sh \
    --config-name inference_thinking_probability.yaml \
    --dataset-name thinking_probability \
    --split tuning

sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time 60 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/natural_language_train_grpo/%j.out \
    scripts/natural_language_train_grpo.sh \
    --config-name grpo_yes_no.yaml \
    --dataset-name thinking_yes_no

sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:4 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time 720 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/natural_language_train_grpo/%j.out \
    scripts/natural-language-train-grpo-multi-gpu.sh \
    --config-name grpo_probability.yaml \
    --dataset-name thinking_probability


sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:4 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --time 600 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/natural_language_train_grpo/%j.out \
    scripts/natural-language-train-grpo-multi-gpu.sh \
    --config-name grpo_yes_no.yaml \
    --dataset-name thinking_yes_no

sbatch\
    --partition cpu \
    --time 360 \
    --mem 16G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=/scratch/users/%u/logs/trackio/%j.out \
    scripts/trackio.sh

bash scripts/trackio.sh

srun -p cpu --time 240 --mem=8G --cpus-per-task=1 --pty /bin/bash -l

srun --partition biomed_a100_gpu --cpus-per-task=1 --gres=gpu:1 --time 20 --mem=16G --pty /bin/bash -l

srun --partition biomed_a100_gpu --cpus-per-task=8 --gres=gpu:1 --time 20 --mem=64G --pty /bin/bash -l
module load cuda/12.2.1-gcc-13.2.0
```

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
        - `data/MEDS_cohort/merge_to_MEDS_cohort`
    - coded data
        - `data/MEDS_cohort/bin_numeric_values`
        - Has age as string rather than numeric and quantiled numeric values
    - natural language data
        - `data/MEDS_cohort/data`
        - Has numeric values as strings appended to codes

Create ACES labels
- `make aces-labels`
- Output is in `data/aces_labels`



Using `uv` to run Python commands

Run tests, linters, and Python CLI tools from the project root using `uv` so the project's locked environment is used. Example:

```sh
uv run pytest -q
```

Prefer `uv run <command>` for all Python commands (tests, CLI tools, downloads, etc.).

uv run meds-tab-describe "MEDS_cohort_dir=data/MEDS_cohort" "input_dir=data/MEDS_cohort/merge_to_MEDS_cohort/train"


uv run meds-tab-cache-task \
    "MEDS_cohort_dir=data/MEDS_cohort" \
    "input_dir=data/MEDS_cohort/merge_to_MEDS_cohort" \
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
