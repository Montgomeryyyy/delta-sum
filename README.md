# mimic-struct-predict

## CREATE Helper commands

Get an interative session
`srun --cpus-per-task=2 --mem=8G -p slam_cpu --time 1-00:00:00 --pty /bin/bash -l`


## Setup

Setup up .env
```
export MIMIC_USER=
export MIMIC_PASSWORD=
export HF_TOKEN=
export SCRATCH_DIR=
```

Set enviroment variables
```
source .env
source scripts/config.sh
```

Install uv. This takes a while due to building flash attention from source. Use H100 so CUDA drivers loaded correctly.
```
sbatch \
    --partition slam_gpu \
    --gres=gpu:1 \
    --constraint=h100 \
    --nodes=1 \
    --ntasks=12 \
    --mem=512G \
    --time 60 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/setup_uv/%j.out \
    scripts/setup_uv.sh
```

### Data Prep

export DEMO=true
or
export DEMO=false

sbatch\
    --partition slam_cpu \
    --time=0:10:00 \
    --mem 2G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN,MIMIC_USER,MIMIC_PASSWORD,DEMO \
    --output=$SCRATCH_DIR/../logs/download_mimic/%j.out \
    scripts/download_mimic.sh

sbatch\
    --partition slam_cpu \
    --time=1:00:00 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=256G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN,DEMO \
    --output=$SCRATCH_DIR/../logs/meds-dataset/%j.out \
    scripts/meds-dataset.sh

## Pre-training

OR
export MODE=natural_language_lora
OR

export MODE=natural_language
export MODEL_NAME=$NL_MODEL_DIR

export MODE=structured
export MODEL_NAME=$STRUCT_MODEL_DIR

```
sbatch\
    --partition slam_cpu \
    --time=0:20:00 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=256G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-tokenize/%j.out \
    scripts/pretrain-tokenize.sh $MODE

sbatch\
    --partition slam_gpu \
    --time=7-00:00:00 \
    --gres=gpu:1 \
    --constraint=h100 \
    --ntasks=1 \
    --cpus-per-task=18 \
    --mem=64G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-train/%j.out \
    scripts/pretrain-train.sh $MODE $MODEL_NAME
```

### Inference

```
bash scripts/calculate-task-prevalences.sh
```

export MODE=natural_language
export MODEL_NAME=$NL_MODEL_DIR
export OUTPUT_DIRECTORY=cpt

export MODE=structured
export MODEL_NAME=$STRUCT_MODEL_DIR
export OUTPUT_DIRECTORY=pt

sbatch\
    --partition slam_cpu \
    --time=4:00:00 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --mem=512G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN,DEMO \
    --output=$SCRATCH_DIR/../logs/create-task-datasets/%j.out \
    scripts/create-task-datasets.sh $MODE $MODEL_NAME

sbatch\
    --partition slam_cpu \
    --time=0:10:00 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=16G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN,DEMO \
    --output=$SCRATCH_DIR/../logs/create-task-datasets/%j.out \
    scripts/create-task-datasets.sh $MODE $MODEL_NAME

sbatch\
    --partition slam_gpu \
    --time=0-12:00:00 \
    --gres=gpu:4 \
    --constraint=l40s \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-generation/%j.out \
    scripts/pretrain-generation-vllm.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split held_out

sbatch\
    --partition slam_gpu \
    --time=2-00:00:00 \
    --gres=gpu:4 \
    --constraint=l40s \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-generation/%j.out \
    scripts/pretrain-generation-hf-small.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split held_out

sbatch\
    --partition biomed_a100_gpu \
    --time=2-00:00:00 \
    --gres=gpu:4 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-generation/%j.out \
    scripts/pretrain-generation-hf.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split held_out

sbatch\
    --partition slam_gpu \
    --time=2-00:00:00 \
    --gres=gpu:1 \
    --constraint=h100 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=64G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-generation/%j.out \
    scripts/pretrain-generation-hf.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split held_out

    scripts/pretrain-generation_vllm.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split tuning

    scripts/pretrain-generation.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split tuning

    scripts/pretrain-generation_vllm.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY --split tuning


## Eval

sbatch\
    --partition slam_cpu \
    --time=1:00:00 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --mem=256G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/evals/%j.out \
    scripts/eval.sh

sbatch\
    --partition slam_cpu \
    --time=1:00:00 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=32G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/eval_metrics/%j.out \
    scripts/eval_metrics.sh

sbatch\
    --partition slam_cpu \
    --time=0:10:00 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=32G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/eval_metrics/%j.out \
    scripts/eval_tokenizer.sh

sbatch\
    --partition slam_cpu \
    --time=4:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/eval_gt_timelines/%j.out \
    scripts/eval_gt_timelines.sh \
    --nl_tokenizer_path /scratch/prj/bhi_ehrfoundationmodels/k2692250/mimic-struct-predict/training/natural_language/pretraining/20260819_155553/final_model \
    --struct_tokenizer_path /scratch/prj/bhi_ehrfoundationmodels/k2692250/mimic-struct-predict/training/structured/pretraining/20260814_191104/final_model \
    --split held_out

sbatch\
    --partition slam_cpu \
    --time=1:00:00 \
    --ntasks=1 \
    --cpus-per-task=32 \
    --mem=64G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/eval_plots/%j.out \
    scripts/eval_plots.sh --nl_tokenizer_path $NL_MODEL_DIR --struct_tokenizer_path $STRUCT_MODEL_DIR


## Linear Probe

export MODE=natural_language
OR
export MODE=structured
OR
export MODE=natural_language_lora

export MODEL_NAME=$HF_MODEL_NAME

export OUTPUT_DIRECTORY=cpt/pt/zero_shot/cpt_lora

sbatch\
    --partition slam_cpu \
    --time=4:00:00 \
    --ntasks=1 \
    --cpus-per-task=64 \
    --mem=512G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/create-task-datasets/%j.out \
    scripts/create-task-datasets.sh $MODE $MODEL_NAME

sbatch\
    --partition slam_gpu \
    --time=3-00:00:00 \
    --gres=gpu:1 \
    --constraint=h100 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/linear-probe-embedding/%j.out \
    scripts/linear-probe-embedding.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY

sbatch\
    --partition slam_cpu \
    --time=1:00:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --chdir $(pwd) \
    --output=$SCRATCH_DIR/../logs/linear-probe-train/%j.out \
    scripts/linear-probe-train.sh $MODE $OUTPUT_DIRECTORY

sbatch\
    --partition slam_cpu \
    --time=0:30:00 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --chdir $(pwd) \
    --output=$SCRATCH_DIR/../logs/linear-probe-inference/%j.out \
    scripts/linear-probe-inference.sh $MODE $OUTPUT_DIRECTORY --split held_out

## MEDS-TAB

sbatch\
    --partition slam_cpu \
    --time=2-00:00:00 \
    --nodes=1 \
    --cpus-per-task=124 \
    --mem=950G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/meds-tab/%j.out \
    scripts/meds-tab.sh

### SFT

sbatch \
    --partition slam_gpu \
    --gres=gpu:2 \
    --constraint=h100 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time 2000 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/natural_language_train_sft/%j.out \
    scripts/natural_language_train_sft_multi_gpu.sh \
    --config-name sft.yaml \
    --dataset-name direct_yes_no

sbatch\
    --partition slam_cpu \
    --cpus-per-task=8 \
    --mem=256G \
    --time=60 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/natural-language-dataset-prompts/%j.out \
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
    --output=$SCRATCH_DIR/../logs/natural_language_inference/%j.out \
    scripts/natural_language_inference.sh \
    --config-name inference_thinking_yes_no.yaml \
    --dataset-name thinking_yes_no \
    --split tuning

bash scripts/natural_language_inference.sh \
    --config-name inference_direct.yaml \
    --dataset-name direct_yes_no \
    --split tuning


### GRPO

sbatch \
    --partition biomed_a100_gpu \
    --gres=gpu:8 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time 2880 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=$SCRATCH_DIR/../logs/natural_language_train_grpo/%j.out \
    scripts/natural-language-train-grpo-multi-gpu.sh \
    --config-name grpo_probability.yaml \
    --dataset-name thinking_probability

sbatch \
    --partition slam_gpu \
    --gres=gpu:2 \
    --constraint=h100 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time 2880 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR \
    --output=$SCRATCH_DIR/../logs/natural_language_train_grpo/%j.out \
    scripts/natural-language-train-grpo-multi-gpu.sh \
    --config-name grpo_probability.yaml \
    --dataset-name thinking_probability


    --config-name grpo_probability.yaml \
    --dataset-name thinking_probability

sbatch\
    --partition slam_gpu \
    --time=1-00:00:00 \
    --gres=gpu:2 \
    --constraint=l40s \
    --ntasks=1 \
    --cpus-per-task=14 \
    --mem=64G \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/pretrain-inference-prompt/%j.out \
    scripts/pretrain-inference-prompt.sh $MODE $MODEL_NAME $OUTPUT_DIRECTORY


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
    --output=$SCRATCH_DIR/../logs/natural_language_inference/%j.out \
    scripts/natural_language_inference.sh \
    --config-name inference_thinking_probability.yaml \
    --dataset-name thinking_probability \
    --split tuning

sbatch \
    --partition slam_gpu \
    --gres=gpu:4 \
    --constraint=l40s \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --time 30 \
    --chdir $(pwd) \
    --export=SCRATCH_DIR,HF_TOKEN \
    --output=$SCRATCH_DIR/../logs/natural_language_inference/%j.out \
    scripts/natural_language_inference.sh \
    --config-name inference_direct.yaml \
    --dataset-name direct_yes_no \
    --split tuning \
    --model-path google/gemma-3-270m-it \
    --lora-adaptor /scratch/users/k2692250/mimic-struct-predict/training/natural_language/sft/direct_yes_no/20260716_221911/final_model


    --model-path google/gemma-3-12b-it


    --lora-adaptor /scratch/users/k2692250/mimic-struct-predict/training/natural_language/sft/direct_yes_no/20260714_153800/checkpoint-100



    --model-path Qwen/Qwen3-8B



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
