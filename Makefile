# DATASET_NAME = mimic_iv
# MIMIC_VERSION = 3.1

DATASET_NAME = mimic-iv-demo
MIMIC_VERSION = 2.2

PHYSIONET_URL    := https://physionet.org/files

MIMICIV_RAW_DIR      := data/raw_input
MIMICIV_PRE_MEDS_DIR := data/pre_MEDS
MIMICIV_MEDS_COHORT_DIR     := data/MEDS_cohort

MEDS_DIR := foresight_r/preprocessing/meds
N_WORKERS          := 1

define check_credentials
	@test -n "$(MIMIC_USER)" || (echo "Error: MIMIC_USER required" && exit 1)
	@test -n "$(MIMIC_PASSWORD)" || (echo "Error: MIMIC_PASSWORD required" && exit 1)
endef

.PHONY: download-mimic-demo
download-mimic:
	$(call check_credentials)
	@mkdir -p $(MIMICIV_RAW_DIR)
	@cd $(MIMICIV_RAW_DIR) && \
		wget -r -N -c -np --user $(MIMIC_USER) --password $(MIMIC_PASSWORD) \
			--no-check-certificate --accept gz,csv --reject "index.html*" \
			$(PHYSIONET_URL)/$(DATASET_NAME)/$(MIMIC_VERSION) && \
		[ -d physionet.org ] && mv physionet.org/files/$(DATASET_NAME)/$(MIMIC_VERSION)/* . && \
		rm -rf physionet.org || true
	@echo "Downloaded MIMIC-IV-demo v$(MIMIC_VERSION) to $(MIMICIV_RAW_DIR)"


.PHONY: meds-dataset
meds-dataset:
	@echo "Creating pre-MEDS dataset from MIMIC-IV..."
	rm -rf $(MIMICIV_PRE_MEDS_DIR)
	uv run foresight_r/preprocessing/meds_mimic/pre_MEDS.py \
		input_dir=$(MIMICIV_RAW_DIR) \
		output_dir=$(MIMICIV_PRE_MEDS_DIR) \
		do_overwrite=true \
		do_copy=true

	@echo "Running MEDS transform pipeline..."
	rm -rf $(MIMICIV_MEDS_COHORT_DIR)
	uv run MEDS_transform-pipeline \
		"./foresight_r/preprocessing/meds_mimic/config/ETL.yaml"\
		--overrides \
		"input_dir=$(MIMICIV_PRE_MEDS_DIR)" \
		"output_dir=$(MIMICIV_MEDS_COHORT_DIR)" \
		"etl_metadata.dataset_name=$(DATASET_NAME)" \
		"etl_metadata.dataset_version=$(MIMIC_VERSION)"
