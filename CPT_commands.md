export MIMIC_USER=
export MIMIC_PASSWORD=
export HF_TOKEN=
export SCRATCH_DIR=./.scratch_dir
source scripts/config.sh


bash scripts/download_mimic.sh --demo

bash scripts/meds-dataset.sh

bash scripts/pretrain-tokenize.sh natural_language

bash scripts/pretrain-tokenize.sh structured

bash scripts/pretrain-train.sh natural_language

bash scripts/pretrain-train.sh structured
