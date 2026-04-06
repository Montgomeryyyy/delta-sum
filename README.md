# mimic-struct-predict


## Implemented

### Data prep

Download mimic
`make download-mimic`
Can comment out for demo or full dataset


## Plan

### Data prep

- ~~Download meds~~
- premeds
- meds pipeline
- aces
    - Need to define tasks

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
