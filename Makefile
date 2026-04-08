N_WORKERS          := 1

# DATASET_NAME = mimic_iv
# MIMIC_VERSION = 3.1

DATASET_NAME = mimic-iv-demo
MIMIC_VERSION = 2.2

PHYSIONET_URL    := https://physionet.org/files

MIMICIV_RAW_DIR      := data/raw_input
MIMICIV_PRE_MEDS_DIR := data/pre_MEDS
MIMICIV_MEDS_COHORT_DIR     := data/MEDS_cohort

MEDS_TAB_INPUT_DIR := data/MEDS_cohort/add_age_rows
MEDS_TAB_OUTPUT_DIR := data/MEDS_tab

NATURAL_LANGUAGE_INPUT_DIR := data/MEDS_cohort/data
NATURAL_LANGUAGE_OUTPUT_DIR := data/natural_language

MEDS_DIR := mimic_struct_predict/preprocessing/meds

ACES_OUTPUT_DIR := data/aces_labels
ACES_DEFINITIONS_DIR := mimic_struct_predict/preprocessing/aces_task_definitions
# ACES_TASKS := $(notdir $(basename $(wildcard $(ACES_DEFINITIONS_DIR)/*.yaml)))
ACES_TASKS := 30_day_readmission




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
	uv run pre_MEDS \
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

.PHONY: train-meds-tab
train-meds-tab:
	uv run meds-tab-describe \
		"input_dir=$(MEDS_TAB_INPUT_DIR)" "output_dir=$(MEDS_TAB_OUTPUT_DIR)"

	@echo "Tabularizing static data"
	uv run meds-tab-tabularize-static \
		"input_dir=$(MEDS_TAB_INPUT_DIR)" "output_dir=$(MEDS_TAB_OUTPUT_DIR)" \
		do_overwrite=False

	@echo "Tabularizing time series..."
	uv run meds-tab-tabularize-time-series \
		"input_dir=$(MEDS_TAB_INPUT_DIR)" "output_dir=$(MEDS_TAB_OUTPUT_DIR)" \
		tabularization.min_code_inclusion_count=10 \
		tabularization.window_sizes=[1d,7d,30d,365d] \
		tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max] \
		do_overwrite=False

	@for task in $(ACES_TASKS); do \
		mkdir -p "$(MEDS_TAB_OUTPUT_DIR)/$$task"; \
		rsync -r "$(MEDS_TAB_OUTPUT_DIR)/metadata/" "$(MEDS_TAB_OUTPUT_DIR)/$$task/metadata"; \
		echo "Running task_specific_caching.py for task: $$task"; \
		uv run meds-tab-cache-task \
			hydra/launcher=joblib \
			"input_dir=$(MEDS_TAB_INPUT_DIR)" "output_dir=$(MEDS_TAB_OUTPUT_DIR)" \
			"input_label_dir=$(ACES_OUTPUT_DIR)/$$task/" "task_name=$$task" do_overwrite=False; \
		echo "Running xgboost for task: $$task"; \
		uv run meds-tab-xgboost \
			"input_dir=$(MEDS_TAB_INPUT_DIR)" "output_dir=$(MEDS_TAB_OUTPUT_DIR)" \
			"output_model_dir=$(MEDS_TAB_OUTPUT_DIR)/models/$$task/" "task_name=$$task" do_overwrite=False \
			"hydra.sweeper.n_trials=1000" "hydra.sweeper.n_jobs=$(N_WORKERS)"; \
	done


### Natural Language ###

.PHONY: create-nl-dataset
create-nl-dataset:
	uv run create_nl_dataset \
		meds_data_dir=$(NATURAL_LANGUAGE_INPUT_DIR) \
		aces_labels_dir=$(ACES_OUTPUT_DIR) \
		output_dir=$(NATURAL_LANGUAGE_OUTPUT_DIR)

.PHONY: create-nl-dataset-no-labels
create-nl-dataset-no-labels:
	uv run create_nl_dataset_no_labels \
		meds_data_dir=$(NATURAL_LANGUAGE_INPUT_DIR) \
		output_dir=$(NATURAL_LANGUAGE_OUTPUT_DIR)