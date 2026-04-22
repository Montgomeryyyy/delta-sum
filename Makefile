N_WORKERS          := 6

# DATASET_NAME = mimic_iv
# MIMIC_VERSION = 3.1

DATASET_NAME = mimic-iv-demo
MIMIC_VERSION = 2.2

PHYSIONET_URL    := https://physionet.org/files

MIMICIV_RAW_DIR      := data/raw_input
MIMICIV_PRE_MEDS_DIR := data/pre_MEDS
MIMICIV_MEDS_COHORT_DIR     := data/MEDS_cohort


NATURAL_LANGUAGE_INPUT_DIR := data/MEDS_cohort_with_headers
NATURAL_LANGUAGE_OUTPUT_DIR := data/natural_language

DATA_DIR := data
TRAINING_DIR := outputs/training
INFERENCE_DIR := outputs/inference

MEDS_DIR := mimic_struct_predict/preprocessing/meds

ACES_OUTPUT_DIR := data/aces_labels
ACES_DEFINITIONS_DIR := mimic_struct_predict/preprocessing/aces_task_definitions
ACES_TASKS := $(notdir $(basename $(wildcard $(ACES_DEFINITIONS_DIR)/*.yaml)))
# ACES_TASKS := 30_day_readmission

NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR := $(INFERENCE_DIR)/natural_language
STRUCTURED_INPUT_DATA_DIR := $(DATA_DIR)/MEDS_cohort_with_headers_and_quantiles


TRACKIO_DIR := $(TRAINING_DIR)/trackio


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

	@echo "Running add headers pipeline..."
	rm -rf $(NATURAL_LANGUAGE_INPUT_DIR)
	INPUT_DIR=$(MIMICIV_MEDS_COHORT_DIR) \
	OUTPUT_DIR=$(NATURAL_LANGUAGE_INPUT_DIR) \
	DATASET_NAME=$(DATASET_NAME) \
	DATASET_VERSION=$(MIMIC_VERSION) \
	uv run MEDS_transform-pipeline \
		pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/add_headers.yaml\

	@echo "Running add quantiles pipeline..."
	rm -rf $(STRUCTURED_INPUT_DATA_DIR)
	INPUT_DIR=$(NATURAL_LANGUAGE_INPUT_DIR) \
	OUTPUT_DIR=$(STRUCTURED_INPUT_DATA_DIR) \
	DATASET_NAME=$(DATASET_NAME) \
	DATASET_VERSION=$(MIMIC_VERSION) \
	uv run MEDS_transform-pipeline \
		pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/add_quantiles.yaml\


.PHONY: aces-labels
aces-labels:
	@rm -rf $(ACES_OUTPUT_DIR)
	for task in $(ACES_TASKS); do \
		uv run aces-cli -m data=sharded \
			config_path=$(ACES_DEFINITIONS_DIR)/$$task.yaml \
			data.standard=meds \
			data=sharded \
			data.root=$(MIMICIV_MEDS_COHORT_DIR)/data \
			"data.shard=$$(uv run expand_shards $(MIMICIV_MEDS_COHORT_DIR)/data)" \
			cohort_dir=$(ACES_OUTPUT_DIR) \
			cohort_name="$$task" \
			-m; \
	done

MEDS_TAB_OUTPUT_DATA_DIR := $(DATA_DIR)/MEDS_tab
MEDS_TAB_OUTPUT_MODEL_DIR := $(TRAINING_DIR)/MEDS_tab

.PHONY: train-meds-tab
train-meds-tab:
# 	@echo "Describing MEDS tabular data..."
# 	uv run meds-tab-describe \
# 		"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
# 		"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)"

# 	@echo "Tabularizing time series..."
# 	uv run meds-tab-tabularize-time-series \
# 		"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
# 		"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)" \
# 		"tabularization.min_code_inclusion_count=10" \
# 		"tabularization.window_sizes=[1d,7d,30d,365d]" \
# 		"tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
# 		"do_overwrite=False"

	@for task in $(ACES_TASKS); do \
		echo "Processing task: $$task"; \
		mkdir -p "$(MEDS_TAB_OUTPUT_DATA_DIR)/$$task"; \
		rsync -r "$(MEDS_TAB_OUTPUT_DATA_DIR)/metadata/" "$(MEDS_TAB_OUTPUT_DATA_DIR)/$$task/metadata"; \
		echo "  -> Running task_specific_caching.py"; \
		uv run meds-tab-cache-task \
			hydra/launcher=joblib \
			"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
			"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)" \
			"input_label_dir=$(ACES_OUTPUT_DIR)/$$task/" \
			"task_name=$$task" \
			"do_overwrite=False"; \
		echo "  -> Running xgboost"; \
		uv run meds-tab-model \
			--multirun \
			model_launcher=xgboost \
			"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
			"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)" \
			"output_model_dir=$(MEDS_TAB_OUTPUT_MODEL_DIR)/$$task/" \
			"task_name=$$task" \
			"do_overwrite=False" \
			"hydra.sweeper.n_trials=15" \
			"hydra.sweeper.n_jobs=$(N_WORKERS)" \
			"tabularization.min_code_inclusion_count=10" \
			"tabularization.window_sizes=$(shell uv run generate-subsets "[1d,30d,365d]")" \
    		"tabularization.aggs=$(shell uv run generate-subsets "[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]")" \
			"delete_below_top_k=3"; \
	done

### Structured ###

STRUCTURED_OUTPUT_DATA_DIR := $(DATA_DIR)/structured
STRUCTURED_PRETRAINING_DATA_DIR := $(STRUCTURED_OUTPUT_DATA_DIR)/pretraining
STRUCTURED_MODEL_DIR := $(TRAINING_DIR)/structured

.PHONY: structured_pretraining_tokenize
structured_pretraining_tokenize:
	uv run structured_pretraining_tokenize \
		dataset_dir=$(STRUCTURED_INPUT_DATA_DIR)/data \
		output_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)

.PHONY: train-structured-pretraining
structured_pretraining:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run structured_pretraining \
	    dataset_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenized_dataset \
		tokenizer_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenizer \
		output_dir=$(STRUCTURED_MODEL_DIR)/pretraining

.PHONY: train-structured-pretraining-multigpu
train-structured-pretraining-multigpu:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run accelerate launch config_file --config_file=./mimic_struct_predict/models/train/structured/pretraining/config/accelerate_multigpu.yaml \
		structured_pretraining \
		dataset_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenized_dataset \
		tokenizer_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenizer \
		output_dir=$(STRUCTURED_MODEL_DIR)/pretraining

.PHONY: structured_create_task_prompts
structured_create_task_prompts:
	uv run structured_create_task_prompts \
		meds_dir=$(STRUCTURED_INPUT_DATA_DIR)/data \
		aces_dir=$(ACES_OUTPUT_DIR) \
		output_dir=$(STRUCTURED_OUTPUT_DATA_DIR)/prompts

.PHONY: structured_inference_generative_rollout
structured_inference_generative_rollout:
	uv run structured_inference_generative_rollout \
		model.pretrained_model_name_or_path=outputs/training/structured/pretraining/20260420_173311/final_model \
		dataset_dir=$(STRUCTURED_OUTPUT_DATA_DIR)/prompts \
		output_dir=$(INFERENCE_DIR)/structured/generative_rollout \
		split=tuning \
		tasks="[los_hosp_3_days, mortality_in_hosp]"

### Natural Language ###


NATURAL_LANGUAGE_DIRECT_DATA_DIR := $(DATA_DIR)/natural_language_direct
NATURAL_LANGUAGE_THINKING_DATA_DIR := $(DATA_DIR)/natural_language_thinking
NATURAL_LANGUAGE_SFT_MODEL_DIR := $(TRAINING_DIR)/natural_language_sft

.PHONY: download-qwen-06B
download-qwen-06B:
	uv run hf auth login
	uv run hf download Qwen/Qwen3-0.6B --local-dir ./pre_trained_models/Qwen3-0.6B

.PHONY: natural_language_create_prompts_direct
natural_language_create_prompts_direct:
	uv run natural_language_create_prompts \
		meds_dir=$(NATURAL_LANGUAGE_INPUT_DIR)/data \
		aces_dir=$(ACES_OUTPUT_DIR) \
		output_dir=$(NATURAL_LANGUAGE_DIRECT_DATA_DIR) \


.PHONY: natural_language_create_prompts_thinking
natural_language_create_prompts_thinking:
	uv run natural_language_create_prompts \
		--config-name create_prompts_thinking.yaml \
		meds_dir=$(NATURAL_LANGUAGE_INPUT_DIR)/data \
		aces_dir=$(ACES_OUTPUT_DIR) \
		output_dir=$(NATURAL_LANGUAGE_THINKING_DATA_DIR)

.PHONY: natural_language_sft
natural_language_sft:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run natural_language_sft \
		dataset_dir=$(NATURAL_LANGUAGE_DIRECT_DATA_DIR)/hf_dataset \
		output_dir=$(NATURAL_LANGUAGE_SFT_MODEL_DIR)


.PHONY: natural_language_inference_direct
natural_language_inference_direct:
	uv run natural_language_inference \
		"dataset_dir=$(NATURAL_LANGUAGE_DIRECT_DATA_DIR)/hf_dataset" \
		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/direct" \
		"split=tuning" \
		tasks="[los_hosp_3_days, mortality_in_hosp]"

.PHONY: natural_language_inference_thinking
natural_language_inference_thinking:
	uv run natural_language_inference \
		--config-name inference_thinking.yaml \
		"dataset_dir=$(NATURAL_LANGUAGE_THINKING_DATA_DIR)/hf_dataset" \
		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/thinking" \
		"split=tuning" \
		"tasks=icu_mortality" \
