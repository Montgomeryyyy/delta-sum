export MIMIC_USER=
export MIMIC_PASSWORD=
export HF_TOKEN=
export SCRATCH_DIR=/scratch$(pwd)
source scripts/config.sh


bash scripts/download_mimic.sh --demo

bash scripts/meds-dataset.sh

bash scripts/natural-language-pretrain-dataset.sh
