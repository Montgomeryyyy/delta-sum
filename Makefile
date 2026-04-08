N_WORKERS          := 1

# DATASET_NAME = mimic_iv
# MIMIC_VERSION = 3.1

DATASET_NAME = mimic-iv-demo
MIMIC_VERSION = 2.2

PHYSIONET_URL    := https://physionet.org/files

MIMICIV_RAW_DIR      := data/raw_input
MIMICIV_PRE_MEDS_DIR := data/pre_MEDS
MIMICIV_MEDS_COHORT_DIR     := data/MEDS_cohort

MEDS_DIR := mimic_struct_predict/preprocessing/meds

ACES_OUTPUT_DIR := data/aces_labels
ACES_DEFINITIONS_DIR := mimic_struct_predict/preprocessing/aces_task_definitions
ACES_TASKS := $(notdir $(basename $(wildcard $(ACES_DEFINITIONS_DIR)/*.yaml)))

define check_credentials
	@test -n "$(MIMIC_USER)" || (echo "Error: MIMIC_USER required" && exit 1)
	@test -n "$(MIMIC_PASSWORD)" || (echo "Error: MIMIC_PASSWORD required" && exit 1)
endef

.PHONY: download-mimic-demo
download-mimic:
	$(call check_credentials)
	@mkdir -p $(MIMICIV_RAW_DIR)
	cd $(MIMICIV_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) \
			--no-check-certificate --accept gz,csv --reject "index.html*" -nH --cut-dirs=3 \
			$(PHYSIONET_URL)/$(DATASET_NAME)/$(MIMIC_VERSION)/
	@echo "Downloaded $(DATASET_NAME) v$(MIMIC_VERSION) to $(MIMICIV_RAW_DIR)"


.PHONY: meds-dataset
meds-dataset:
	@echo "Creating pre-MEDS dataset from MIMIC-IV..."
	rm -rf $(MIMICIV_PRE_MEDS_DIR)
	uv run mimic_struct_predict/preprocessing/meds/pre_MEDS.py \
		input_dir=$(MIMICIV_RAW_DIR) \
		output_dir=$(MIMICIV_PRE_MEDS_DIR) \
		do_overwrite=true \
		do_copy=true

	@echo "Running MEDS transform pipeline..."
	rm -rf $(MIMICIV_MEDS_COHORT_DIR)
	INPUT_DIR=$(MIMICIV_PRE_MEDS_DIR) \
	OUTPUT_DIR=$(MIMICIV_MEDS_COHORT_DIR) \
	DATASET_NAME=$(DATASET_NAME) \
	DATASET_VERSION=$(MIMIC_VERSION) \
	uv run MEDS_transform-pipeline \
		pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/ETL.yaml\


.PHONY: aces-labels
aces-labels:
	@rm -rf $(ACES_OUTPUT_DIR)
	for task in $(ACES_TASKS); do \
		uv run aces-cli -m data=sharded \
			config_path=$(ACES_DEFINITIONS_DIR)/$$task.yaml \
			data.standard=meds \
			data=sharded \
			data.root=$(MIMICIV_MEDS_COHORT_DIR)/order_by_code_prefix \
			"data.shard=$$(uv run expand_shards $(MIMICIV_MEDS_COHORT_DIR)/order_by_code_prefix)" \
			cohort_dir=$(ACES_OUTPUT_DIR) \
			cohort_name="$$task" \
			-m; \
	done