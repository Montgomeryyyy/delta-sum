N_WORKERS          := 2

# MIMIC_VERSION = 3.1

DATASET_NAME = mimic-iv-demo
DATASET_NAME_ED = mimic-iv-ed-demo
MIMIC_VERSION = 2.2

PHYSIONET_URL    := https://physionet.org/files

MIMICIV_RAW_DIR      := data/raw_input
MIMICIV_PRE_MEDS_DIR := data/pre_MEDS
MIMICIV_MEDS_COHORT_DIR := data/MEDS_cohort


NATURAL_LANGUAGE_DATA_DIR := data/MEDS_cohort_with_headers
NATURAL_LANGUAGE_OUTPUT_DIR := data/natural_language

DATA_DIR := data
TRAINING_DIR := outputs/training
INFERENCE_DIR := outputs/inference

MEDS_DIR := mimic_struct_predict/preprocessing/meds

ACES_OUTPUT_DIR := data/aces_labels
ACES_DEFINITIONS_DIR := mimic_struct_predict/preprocessing/aces_task_definitions
ACES_TASKS := $(notdir $(basename $(wildcard $(ACES_DEFINITIONS_DIR)/*.yaml)))
# ACES_TASKS := ed_to_hospital_admission

NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR := $(INFERENCE_DIR)/natural_language
STRUCTURED_INPUT_DATA_DIR := $(DATA_DIR)/MEDS_cohort_with_headers_and_quantiles


TRACKIO_DIR := $(TRAINING_DIR)/trackio


.PHONY: show-trackio
show-trackio:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run trackio show

define check_credentials
	@test -n "$(MIMIC_USER)" || (echo "Error: MIMIC_USER required (use 'export MIMIC_USER=your_username')" && exit 1)
	@test -n "$(MIMIC_PASSWORD)" || (echo "Error: MIMIC_PASSWORD required (use 'export MIMIC_PASSWORD=your_password')" && exit 1)
endef

.PHONY: download-mimic
download-mimic:
	$(call check_credentials)
	@mkdir -p $(MIMICIV_RAW_DIR)
	cd $(MIMICIV_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) \
			--no-check-certificate --accept gz,csv --reject "index.html*" -nH --cut-dirs=3 \
			$(PHYSIONET_URL)/$(DATASET_NAME)/$(MIMIC_VERSION)/
	@echo "Downloaded $(DATASET_NAME) v$(MIMIC_VERSION) to $(MIMICIV_RAW_DIR)"

	cd $(MIMICIV_RAW_DIR) && \
		wget -r -N -c -np --user "$$MIMIC_USER" --password "$$MIMIC_PASSWORD" \
			--no-check-certificate --accept gz,csv --reject "index.html*" -nH --cut-dirs=3 \
			$(PHYSIONET_URL)/$(DATASET_NAME_ED)/$(MIMIC_VERSION)/
	@echo "Downloaded $(DATASET_NAME_ED) v$(MIMIC_VERSION) to $(MIMICIV_RAW_DIR)"


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
	rm -rf $(NATURAL_LANGUAGE_DATA_DIR)
	INPUT_DIR=$(MIMICIV_MEDS_COHORT_DIR) \
	OUTPUT_DIR=$(NATURAL_LANGUAGE_DATA_DIR) \
	DATASET_NAME=$(DATASET_NAME) \
	DATASET_VERSION=$(MIMIC_VERSION) \
	uv run MEDS_transform-pipeline \
		pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/add_headers.yaml\

	@echo "Running add quantiles pipeline..."
	rm -rf $(STRUCTURED_INPUT_DATA_DIR)
	INPUT_DIR=$(NATURAL_LANGUAGE_DATA_DIR) \
	OUTPUT_DIR=$(STRUCTURED_INPUT_DATA_DIR) \
	DATASET_NAME=$(DATASET_NAME) \
	DATASET_VERSION=$(MIMIC_VERSION) \
	uv run MEDS_transform-pipeline \
		pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/add_quantiles.yaml\


.PHONY: aces-dataset
aces-dataset:
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

# Based on https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/MIMICIV_TUTORIAL/task_tabularize_meds.sh
.PHONY: meds-tab-train
meds-tab-train:
	@echo "Describing MEDS tabular data..."
	uv run meds-tab-describe \
		"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
		"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)"

	@for task in $(ACES_TASKS); do \
		echo "Processing task: $$task"; \
		mkdir -p "$(MEDS_TAB_OUTPUT_DATA_DIR)/training/$$task"; \
		rsync -r "$(MEDS_TAB_OUTPUT_DATA_DIR)/metadata/" "$(MEDS_TAB_OUTPUT_DATA_DIR)/training/$$task/metadata"; \
		echo "  -> Running task_specific_tabularization"; \
		uv run meds-tab-tabularize-time-series \
			--multirun \
			worker="range(0,$(N_WORKERS))" \
			hydra/launcher=joblib \
			"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
			"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)/training/$$task" \
			"input_label_dir=$(ACES_OUTPUT_DIR)/$$task/"; \
			"do_overwrite=False" \
			"tabularization.min_code_inclusion_count=10" \
			"tabularization.window_sizes=[1d,7d,30d,365d]" \
			"tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
		echo "  -> Running xgboost"; \
		uv run meds-tab-model \
			--multirun \
			model_launcher=xgboost \
			"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/data" \
			"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)/training/$$task" \
			"input_tabularized_cache_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)/training/$$task/tabularize/" \
        	"input_label_cache_dir=$(ACES_OUTPUT_DIR)/$$task/" \
			"output_model_dir=$(MEDS_TAB_OUTPUT_MODEL_DIR)/$$task/" \
			"task_name=$$task" \
			"do_overwrite=False" \
			"hydra.sweeper.n_trials=4" \
			"hydra.sweeper.n_jobs=$(N_WORKERS)" \
			"tabularization.min_code_inclusion_count=10" \
			"tabularization.window_sizes=$(shell uv run generate-subsets "[1d,7d,30d,365d]")" \
    		"tabularization.aggs=$(shell uv run generate-subsets "[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]")" \
			"delete_below_top_k=1"; \
	done


.PHONY: meds-tab-inference
meds-tab-inference:
	mkdir -p $(MIMICIV_MEDS_COHORT_DIR)/held_out
	cp -r $(MIMICIV_MEDS_COHORT_DIR)/data/held_out $(MIMICIV_MEDS_COHORT_DIR)/held_out/held_out

	uv run meds_tab_inference \
		"model_dir=$(TRAINING_DIR)/MEDS_tab" \
		"input_dir=$(MIMICIV_MEDS_COHORT_DIR)/held_out" \
		"output_dir=$(MEDS_TAB_OUTPUT_DATA_DIR)/inference" \
		"aces_dir=$(ACES_OUTPUT_DIR)" \
		"prediction_dir=$(INFERENCE_DIR)/MEDS_tab" \
		"split=tuning"


### Structured ###

# STRUCTURED_OUTPUT_DATA_DIR := $(DATA_DIR)/structured
# STRUCTURED_PRETRAINING_DATA_DIR := $(STRUCTURED_OUTPUT_DATA_DIR)/pretraining
# STRUCTURED_MODEL_DIR := $(TRAINING_DIR)/structured
# STRUCTURED_BEST_PRETRAINED_MODEL_DIR := $(STRUCTURED_MODEL_DIR)/pretraining/20260514_185803/final_model

# .PHONY: structured-pretraining-tokenize
# structured-dataset-pretraining:
# 	uv run pre_training_tokenize \
# 		--config-name structured.yaml \
# 		dataset_dir=$(STRUCTURED_INPUT_DATA_DIR)/data \
# 		output_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)

# .PHONY: structured-train-pretraining
# structured-train-pretraining:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run pre_training_train \
# 		--config-name full.yaml \
# 		model.random_init=true \
# 	    dataset_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		tokenizer_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenizer \
# 		output_dir=$(STRUCTURED_MODEL_DIR)/pretraining

# .PHONY: structured-dataset-prompts
# structured-dataset-prompts:
# 	uv run structured_create_task_prompts \
# 		meds_dir=$(STRUCTURED_INPUT_DATA_DIR)/data \
# 		aces_dir=$(ACES_OUTPUT_DIR) \
# 		output_dir=$(STRUCTURED_OUTPUT_DATA_DIR)/prompts

# .PHONY: structured-train-pretraining-multigpu
# structured-train-pretraining-multigpu:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run accelerate launch config_file --config_file=./mimic_struct_predict/models/train/structured/pretraining/config/accelerate_multigpu.yaml \
# 		pre_training_train \
# 		dataset_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		tokenizer_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenizer \
# 		output_dir=$(STRUCTURED_MODEL_DIR)/pretraining

# .PHONY: structured-train-embedding-and-lr
# structured-train-embedding-and-lr:
# 	uv run embedding_and_lr_inference_embedding_only \
# 		model.pretrained_model_name_or_path=$(STRUCTURED_BEST_PRETRAINED_MODEL_DIR) \
# 		dataset_dir=$(STRUCTURED_OUTPUT_DATA_DIR)/prompts \
# 		output_dir=$(STRUCTURED_MODEL_DIR)/embedding_lr/embeddings \
# 		splits="[train,tuning]" \

# 	for task_dir in $(STRUCTURED_MODEL_DIR)/embedding_lr/embeddings/*/; do \
# 		task=$$(basename $$task_dir); \
# 		echo "Training embedding logistic regression for task: $$task"; \
# 		uv run embedding_and_lr_train --multirun \
# 			task_dir=$$task_dir \
# 			output_dir=$(STRUCTURED_MODEL_DIR)/embedding_lr/models/$$task; \
# 	done

# .PHONY: structured-inference-embedding-and-lr
# structured-inference-embedding-and-lr:
# 	uv run embedding_and_lr_inference \
# 		model.pretrained_model_name_or_path=$(STRUCTURED_BEST_PRETRAINED_MODEL_DIR) \
# 		dataset_dir=$(STRUCTURED_OUTPUT_DATA_DIR)/prompts \
# 		output_dir=$(INFERENCE_DIR)/structured/embedding_and_lr \
# 		lr_model_dir=$(STRUCTURED_MODEL_DIR)/embedding_lr/models \
# 		split=tuning

# .PHONY: structured-inference-generative-rollout
# structured-inference-generative-rollout:
# 	uv run structured_inference_generative_rollout \
# 		model.pretrained_model_name_or_path=$(STRUCTURED_BEST_PRETRAINED_MODEL_DIR) \
# 		dataset_dir=$(STRUCTURED_OUTPUT_DATA_DIR)/prompts \
# 		output_dir=$(INFERENCE_DIR)/structured/generative_rollout \
# 		split=tuning \

# .PHONY: structured-train-pretraining-tiny
# structured-train-pretraining-tiny:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run pre_training_train \
# 		--config-name tiny_model.yaml \
# 	    dataset_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		tokenizer_dir=$(STRUCTURED_PRETRAINING_DATA_DIR)/tokenizer \
# 		output_dir=$(STRUCTURED_MODEL_DIR)/pretraining_tiny

### Natural Language ###

NATURAL_LANGUAGE_TASK_EHR_TEXT_DIR := $(DATA_DIR)/natural_language/task_ehr_texts
NATURAL_LANGUAGE_PROMPT_DATA_DIR := $(DATA_DIR)/natural_language/prompts
NATURAL_LANGUAGE_THINKING_DATA_DIR := $(DATA_DIR)/natural_language/thinking
NATURAL_LANGUAGE_GEMINI_DATA_DIR := $(DATA_DIR)/natural_language/gemini
NATURAL_LANGUAGE_MODEL_DIR := $(TRAINING_DIR)/natural_language
NATURAL_LANGUAGE_SFT_MODEL_DIR := $(NATURAL_LANGUAGE_SFT_MODEL_DIR)/direct/sft

NATURAL_LANGUAGE_PRETRAINING_DATA_DIR := $(DATA_DIR)/natural_language/pretraining

.PHONY: natural-language-dataset-prompts
natural-language-dataset-prompts:
	uv run natural_language_create_task_ehr_texts \
		meds_dir=$(NATURAL_LANGUAGE_DATA_DIR)/data \
		aces_dir=$(ACES_OUTPUT_DIR) \
		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \

# 	# 	uv run natural_language_create_model_prompts \
# # 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# # 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/direct_yes_no \

	uv run natural_language_create_model_prompts \
		tokenization.enable_thinking=true \
		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_yes_no \

# 	uv run natural_language_create_model_prompts \
# 		tokenization.output_format=probability \
# 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/direct_probability \

# 	uv run natural_language_create_model_prompts \
# 		tokenization.output_format=probability \
# 		tokenization.enable_thinking=true \
# 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_probability \

.PHONY: natural-language-train-grpo-yes-no
natural-language-train-grpo-yes-no:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run natural_language_grpo \
		--config-name grpo_yes_no.yaml \
		dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_yes_no \
		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/grpo_yes_no

.PHONY: natural-language-train-grpo-probability
natural-language-train-grpo-probability:
	TRACKIO_DIR=$(TRACKIO_DIR) \
	uv run natural_language_grpo \
		--config-name grpo_probability.yaml \
		dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_probability \
		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/grpo_probability

.PHONY: natural-language-inference-thinking-probability
natural-language-inference-thinking-probability:
	uv run natural_language_inference \
		--config-name inference_thinking_probability.yaml \
		"dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_probability" \
		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/thinking_probability_grpo" \
		"split=tuning"

.PHONY: natural-language-inference-thinking-yes-no
natural-language-inference-thinking-yes-no:
	uv run natural_language_inference \
		--config-name inference_thinking_yes_no.yaml \
		"dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_yes_no" \
		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/thinking_yes_no" \
		"split=tuning"

.PHONY: natural-language-inference-thinking-probability-grpo
natural-language-inference-thinking-probability-grpo:
	uv run natural_language_inference \
		--config-name inference_thinking_probability.yaml \
		model.pretrained_model_name_or_path=$(NATURAL_LANGUAGE_MODEL_DIR)/grpo_probability/20260520_125506/final_model \
		"dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_probability" \
		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/thinking_probability_grpo" \
		"split=tuning"

.PHONY: natural-language-inference-thinking-yes-no-grpo
natural-language-inference-thinking-yes-no-grpo:
	uv run natural_language_inference \
		--config-name inference_thinking_yes_no.yaml \
		model.pretrained_model_name_or_path=$(NATURAL_LANGUAGE_MODEL_DIR)/grpo_yes_no/20260520_120546/final_model \
		"dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_yes_no" \
		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/thinking_yes_no_grpo" \
		"split=tuning"

# .PHONY: download-qwen-06B
# download-qwen-06B:
# 	uv run hf auth login
# 	uv run hf download Qwen/Qwen3-0.6B --local-dir ./pre_trained_models/Qwen3-0.6B

# .PHONY: structured-pretraining-tokenize
# natural-language-dataset-pretraining:
# 	uv run pre_training_tokenize \
# 		--config-name natural_language.yaml \
# 		dataset_dir=$(STRUCTURED_INPUT_DATA_DIR)/data \
# 		output_dir=$(NATURAL_LANGUAGE_PRETRAINING_DATA_DIR)

# .PHONY: natural-language-train-pretraining-scratch
# natural-language-train-pretraining-scratch:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run pre_training_train \
# 		--config-name full.yaml \
# 		model.random_init=true \
# 	    dataset_dir=$(NATURAL_LANGUAGE_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/pretraining

# .PHONY: natural-language-train-pretraining-full
# natural-language-train-pretraining-full:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run pre_training_train \
# 		--config-name full.yaml \
# 	    dataset_dir=$(NATURAL_LANGUAGE_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/pretraining



# .PHONY: natural-language-train-embedding-and-lr
# natural-language-train-embedding-and-lr:
# 	uv run embedding_and_lr_inference_embedding_only \
# 		model.pretrained_model_name_or_path=$(MODEL_DIR) \
# 		dataset_dir=$(NATURAL_LANGUAGE_THINKING_DATA_DIR) \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/embedding_lr/embeddings \
# 		splits="[train,tuning]" \

# 	for task_dir in $(NATURAL_LANGUAGE_MODEL_DIR)/embedding_lr/embeddings/*/; do \
# 		task=$$(basename $$task_dir); \
# 		echo "Training embedding logistic regression for task: $$task"; \
# 		uv run embedding_and_lr_train --multirun \
# 			task_dir=$$task_dir \
# 			output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/embedding_lr/models/$$task; \
# 	done

# .PHONY: natural-language-inference-embedding-and-lr
# natural-language-inference-embedding-and-lr:
# 	uv run embedding_and_lr_inference \
# 		model.pretrained_model_name_or_path='./pre_trained_models/Qwen3-0.6B' \
# 		dataset_dir=$(NATURAL_LANGUAGE_OUTPUT_DATA_DIR)/prompts \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/embedding_lr/embeddings \
# 		lr_model_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/embedding_lr/models \
# 		split=tuning






# # 	uv run natural_language_create_model_prompts \
# # 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# # 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/direct_yes_no \

# # 	uv run natural_language_create_model_prompts \
# # 		tokenization.enable_thinking=true \
# # 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# # 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_yes_no \

# 	uv run natural_language_create_model_prompts \
# 		tokenization.output_format=probability \
# 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/direct_probability \

# # 	uv run natural_language_create_model_prompts \
# # 		tokenization.output_format=probability \
# # 		tokenization.enable_thinking=true \
# # 		data_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/ehr_texts \
# # 		output_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/thinking_probability \

# .PHONY: natural-language-train-sft
# natural-language-train-sft:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run natural_language_sft \
# 		dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/direct_yes_no \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/sft

# .PHONY: natural-language-train-sft-brier
# natural-language-train-sft-brier:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run natural_language_sft \
# 		brier_loss=true \
# 		dataset_dir=$(NATURAL_LANGUAGE_PROMPT_DATA_DIR)/direct_probability \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/brier_sft




# .PHONY: natural-language-train-pretraining-lora
# natural-language-train-pretraining-lora:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run pre_training_train \
# 		--config-name lora.yaml \
# 	    dataset_dir=$(NATURAL_LANGUAGE_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/pretraining

# .PHONY: natural-language-train-pretraining-tiny
# natural-language-train-pretraining-tiny:
# 	TRACKIO_DIR=$(TRACKIO_DIR) \
# 	uv run pre_training_train \
# 		--config-name tiny_model.yaml \
# 	    dataset_dir=$(NATURAL_LANGUAGE_PRETRAINING_DATA_DIR)/tokenized_dataset \
# 		output_dir=$(NATURAL_LANGUAGE_MODEL_DIR)/pretraining




# .PHONY: natural-language-dataset-prompts-thinking
# natural-language-dataset-prompts-thinking:
# 	uv run natural_language_create_model_prompts \
# 		--config-name create_prompts_thinking.yaml \
# 		data_dir=$(NATURAL_LANGUAGE_RAW_DATA_DIR) \
# 		output_dir=$(NATURAL_LANGUAGE_THINKING_DATA_DIR) \
# 		splits_to_process="[train,tuning]" \

# .PHONY: natural-language-dataset-prompts-gemini
# natural-language-dataset-prompts-gemini:
# 	uv run natural_language_create_model_prompts \
# 		--config-name create_prompts_gemini.yaml \
# 		data_dir=$(NATURAL_LANGUAGE_RAW_DATA_DIR) \
# 		output_dir=$(NATURAL_LANGUAGE_GEMINI_DATA_DIR)



# .PHONY: natural-language-inference-sft
# natural-language-inference-sft:
# 	uv run natural_language_inference \
# 		"model.pretrained_model_name_or_path=$(NATURAL_LANGUAGE_SFT_MODEL_DIR)/20260427_152027/final_model" \
# 		"dataset_dir=$(NATURAL_LANGUAGE_DIRECT_DATA_DIR)" \
# 		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/direct_sft" \
# 		"split=tuning"

# .PHONY: natural-language-inference-direct
# natural-language-inference-direct:
# 	uv run natural_language_inference \
# 		"dataset_dir=$(NATURAL_LANGUAGE_DIRECT_DATA_DIR)" \
# 		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/direct" \
# 		"split=tuning"

# .PHONY: natural-language-inference-gemini
# natural-language-inference-gemini:
# 	uv run natural_language_inference_gemini \
# 		"dataset_dir=$(NATURAL_LANGUAGE_GEMINI_DATA_DIR)" \
# 		"output_dir=$(NATURAL_LANGUAGE_INFERENCE_OUTPUT_DIR)/gemini" \
# 		"split=tuning"
