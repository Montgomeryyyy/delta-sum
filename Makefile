N_WORKERS          := 1

# DATASET_NAME = mimic_iv
# MIMIC_VERSION = 3.1

DATASET_NAME = mimic-iv-demo
MIMIC_VERSION = 2.2

PHYSIONET_URL    := https://physionet.org/files

MIMICIV_RAW_DIR      := data/raw_input
MIMICIV_PRE_MEDS_DIR := data/pre_MEDS
MIMICIV_MEDS_COHORT_DIR     := data/MEDS_cohort

MEDS_TAB_INPUT_DIR := data/MEDS_cohort/merge_to_MEDS_cohort
MEDS_TAB_OUTPUT_DIR := data/MEDS_tab

NATURAL_LANGUAGE_INPUT_DIR := data/MEDS_cohort/data
NATURAL_LANGUAGE_OUTPUT_DIR := data/natural_language

STRUCTURED_INPUT_DIR := data/MEDS_cohort/bin_numeric_values
STRUCTURED_OUTPUT_DIR := data/structured

MEDS_DIR := mimic_struct_predict/preprocessing/meds

ACES_OUTPUT_DIR := data/aces_labels
ACES_DEFINITIONS_DIR := mimic_struct_predict/preprocessing/aces_task_definitions
# ACES_TASKS := $(notdir $(basename $(wildcard $(ACES_DEFINITIONS_DIR)/*.yaml)))
ACES_TASKS := 30_day_readmission

LLM_INFERENCE_OUTPUT_DIR := outputs/llm_inference

TRAINING_DIR := outputs/training
TRACKIO_DIR := $(TRAINING_DIR)/trackio

STRUCTURED_PRETRAINING_DIR := $(TRAINING_DIR)/structured_pretraining
STRUCTURED_PRETRAINING_DATA_DIR := data/structured_pretraining
NL_MULTITASK_SFT_DIR := $(TRAINING_DIR)/natural_language_multitask_sft

.PHONY: show-trackio
show-trackio:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run trackio show

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

### Structured ###

.PHONY: tokenize-structured-pretraining
tokenize-structured-pretraining:
	uv run tokenize_structured_pretraining_dataset \
		dataset_dir=$(STRUCTURED_INPUT_DIR) \
		output_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)

.PHONY: train-structured-pretraining
train-structured-pretraining:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run train_structured_pretraining \
	    dataset_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenized_dataset \
		tokenizer_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenizer \
		output_dir=$(STRUCTURED_PRETRAINING_DIR)

.PHONY: train-structured-pretraining-multigpu
train-structured-pretraining-multigpu:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run accelerate launch config_file --config_file=./mimic_struct_predict/models/train/structured/pretraining/config/accelerate_multigpu.yaml \
		train_structured_pretraining \
		tokenizer_dir=$(STRUCTURED_PRETRAINING_DIR)/tokenizer \
		output_dir=$(STRUCTURED_PRETRAINING_DIR)


### Natural Language ###

.PHONY: download-qwen-06B
download-qwen-06B:
	uv run hf auth login
	uv run hf download Qwen/Qwen3-0.6B --local-dir ./pre_trained_models/Qwen3-0.6B

.PHONY: tokenize-natural-language-multitask-sft
tokenize-natural-language-multitask-sft:
	uv run tokenize_natural_language_multitask_sft \
		meds_dir=$(NATURAL_LANGUAGE_INPUT_DIR) \
		aces_dir=$(ACES_OUTPUT_DIR) \
		output_dir=$(NL_MULTITASK_SFT_DIR)/tokenized_dataset

.PHONY: train-natural-language-multitask-sft
train-natural-language-multitask-sft:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run train_natural_language_multitask_sft \
		dataset_dir=$(NL_MULTITASK_SFT_DIR)/tokenized_dataset \
		output_dir=$(NL_MULTITASK_SFT_DIR)

.PHONY: llm-inference-direct
llm-inference-direct:
	uv run llm_inference \
		"dataset_dir=$(NATURAL_LANGUAGE_OUTPUT_DIR)" \
		"output_dir=$(LLM_INFERENCE_OUTPUT_DIR)"/direct \
		"split=tuning"

.PHONY: llm-inference-thinking
llm-inference-thinking:
	uv run llm_inference \
		--config-name llm_inference_thinking.yaml \
		"dataset_dir=$(NATURAL_LANGUAGE_OUTPUT_DIR)" \
		"output_dir=$(LLM_INFERENCE_OUTPUT_DIR)"/thinking \
		"split=tuning" \
		"tasks=icu_mortality" \
