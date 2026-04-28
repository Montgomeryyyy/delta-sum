# MIMIC Struct Predict

## Background

- Increasing interest in applying modern AI techniques to EHR prediction tasks. Broadly the approaches fall into
    - Tabular (MEDS-Tab, XGBoost)
    - Structured (Foresight, ETHOS, COMET)
    - Natural Language (LLMs)
- Considerable interest has emerged regarding the ability to adapt LLM training methods to produce zero shot predictions
- Due to strict medical privacy constraints and the necessity for large-scale data, most models are evaluated on proprietary datasets or employ bespoke preprocessing pipelines, severely hindering head-to-head comparisons
- However, open evaluation and benchmarking is essential in order to enable iterative methodological advancements.

We contribute
- Fixed MEDS based pre-processing pipeline for MIMIC-IV that generates identically sourced cohorts and time-aligned features, allowing for fair comparison of representation strategies.
    - The use MEDS and ACES means that all training and inference pipelines are transportable to private datasets once converted to MEDS format
- 7 operational outcome tasks that are defined equivalently for all modalities using ACES
- Using the framework we assess the following state of the art modelling approaches:
    - Tabular
        - MEDS-Tab (XGBoost)
    - Structured Code Pretraining
        - Generative Rollout
        - Embeddings + Logistic Regression
    - Natural Language
        - Local (Qwen TBD)
            - Zero Shot Direct
            - Zero Shot Thinking
            - Supervised Fine Tunes
        - Frontier (Gemini 3.1 Pro)
            - Thinking (WIP)

We evaluate
    - Performance: AUROC, AUPRC
    - Calibration: Brier Score, ECE
    - Inference Demand: Throughput, cost
    - Reliability: Format Adherence

## Methods

### Pre-processing

Start with MIMIC-IV vTBC

#### Premed
**Admissions**
- Map gender (e.g., "F" -> "Female")
- Calculate age at admission

**Diagnoses ICD & Procedures ICD**
- Add admission discharge time
- Join dictionary to add `long_title`
- Convert `chartdate` to end-of-day timestamp

**DRG Codes**
- Map severity and mortality levels to text (e.g., 1 -> "MINOR")
- Add admission discharge time

**HCPCS Events**
- Join dictionary to add procedure descriptions
- Convert `chartdate` to end-of-day timestamp

**Lab Events**
- Join dictionary to add `fluid` and `label`
- Fill null `valueuom` with empty strings
- Clear text `value` if numeric `valuenum` is present

**OMR**
- Split blood pressure into separate systolic and diastolic rows
- Clean result names and cast BMI, height, and weight to floats
- Convert `chartdate` to end-of-day timestamp

**Patients**
- Aggregate and join minimum `deathtime` from admissions
- Fallback missing `deathtime` to `dod` (converted to end-of-day timestamp)

**ICU Chart Events**
- Join dictionary to add `label`
- Fill null `valueuom` with empty strings
- Clear text `value` if numeric `valuenum` is present

**ICU Procedure Events**
- Join dictionary to add `label`

**ICU Input Events**
- Join dictionary to add `label`
- Fill null `rateuom` and `amountuom` with empty strings

**ICU Output Events**
- Join dictionary to add `label`
- Fill null `valueuom` with empty strings

#### ETL

Takes MIMIC data and converts to a format of ("subject_id", "timestamp", "code", "numeric_value"). This mapping is controlled by a schema. Each code is also given a category prefix. Values from different rows are joined by "//" delimiter

| Category | Timestamp Column | Code (Full Path) |
| :--- | :--- | :--- |
| **Emergency Department** | edregtime | Emergency Department // Registration |
|  | edouttime | Emergency Department // Out |
| **Hospital Admission** | admittime | Hospital Admission // Type // col(admission_type) // From // col(admission_location) |
| **Patient Demographics** | admittime | Patient Demographics // Insurance // col(insurance) |
|  | admittime | Patient Demographics // Language // col(language) |
|  | admittime | Patient Demographics // Marital Status // col(marital_status) |
|  | admittime | Patient Demographics // Race // col(race) |
|  | admittime | Patient Demographics // Gender // col(gender) |
|  | admittime | Patient Demographics // Age |
| **Hospital Discharge** | dischtime | Hospital Discharge // To // col(discharge_location) |
| **Diagnosis** | hadm_discharge_time | Diagnosis // col(long_title) |
| **Diagnosis Related Group** | hadm_discharge_time | Diagnosis Related Group // col(description) // Severity // col(drg_severity) // Risk Of Mortality // col(drg_mortality) |
| **Medication Administration** | charttime | Medication Administration // col(medication) // col(event_txt) |
| **HCPCS Billing Item** | chartdate | HCPCS Billing Item // col(long_description) |
| **Laboratory Test** | charttime | Laboratory Test // col(fluid) // col(label) // Priority // col(priority) // Value // col(valueuom) // col(value) |
| **Clinical Observation** | chartdate | Clinical Observation // col(result_name) |
| **Death** | dod | Death // Died |
| **Pharmacy Prescription** | starttime | Pharmacy Prescription // Start // col(medication) // Route // col(route) // Frequency // col(frequency) // Doses Per 24 Hrs // col(doses_per_24_hrs) |
|  | stoptime | Pharmacy Prescription // Stop // col(medication) |
| **Hospital Procedure** | chartdate | Hospital Procedure // col(long_title) |
| **Hospital Transfer** | intime | Hospital Transfer // Type // col(eventtype) // To // col(careunit) |
| **Icu Admission** | intime | Icu Admission // To // col(first_careunit) |
| **Icu Discharge** | outtime | Icu Discharge // From // col(last_careunit) |
| **Icu Observation** | charttime | Icu Observation // col(label) // Value // col(valueuom) // col(value) |
| **Icu Procedure** | starttime | Icu Procedure // Start // col(label) |
|  | endtime | Icu Procedure // End // col(label) |
| **Icu Input** | starttime | Icu Input // Start // col(label) // Order Category // col(ordercategorydescription) // Rate // col(rateuom) |
|  | endtime | Icu Input // End // col(label) // Order Category // col(ordercategorydescription) // Status // col(statusdescription) // Amount // col(amountuom) |
|  | starttime | Icu Observation // Patient Weight // KG |
| **Icu Output** | charttime | Icu Output // col(label) // col(valueuom) |

Then apply the following transformations

- Split into train/test/val (80%/10%/10%) on subject_ids
- Regex replacement
    - Clean up repeated // or // at end of code
    - Replace UNK with UNKNOWN
- Title case all codes
- Order codes that happen at same timestamp (priority 1st to last)
    - Birth
        - "Birth"
    - Admissions
        - "Emergency Department"
        - "Hospital Transfer" # Includes ED->Hospital transfers
        - "Hospital Admission"
        - "Patient Demographics" # Recorded at admission time
        - "Icu Admission"
    - Lab results and observations
        - "Laboratory Test"
        - "Clinical Observation"
        - "Icu Observation"
        - "Icu Output"
    - Interventions and treatments
        - "Pharmacy Prescription"
        - "Medication Administration"
        - "Icu Medication"
        - "Icu Input"
        - "Hospital Procedure"
        - "Icu Procedure"
    - Death (discharge and billing happens after death)
        - "Death"
    Discharges
        - "Icu Discharge"
        - "Hospital Discharge"
    Billing codes
        - "Diagnosis"
        - "Diagnosis Related Group"
        - "Hcpcs Billing Item"

The output of this ETL pipeline is used for the tabular modelling

#### Add headers

To reduce repetition and allow for explicit temporal modelling in the natural language and structured modelling approaches we extract headers. We add 3 types of header

**Category Header**
- The category of a code is defined by the part before the first "//"
    - e.g. Emergency Department // Registration -> Category == Emergency Department
- If 2 adjacent events happen on different dates, times or category a category header is inserted
    - E.g. (1, 1, 2000, 12, 00: Cat A//Event A), (1, 1, 2000, 12, 00: Cat B//Event D) -> Insert (1, 1, 2000, 12, 10: #### Category//Cat B)


**Time Header**
- If the 2 adjacent events happen on different dates and or time a time token is added with the prefix "### Time""
    - E.g. (1, 1, 2000, 12, 00: Event A), (1, 1, 2000, 12, 10: Event B) -> Insert (1, 1, 2000, 12, 10: ### Time//12h//00m)
- If the timestamp is end of day (from being a chart event), code "### Time//Unspecified" is used

**Delta Header**

- If the 2 adjacent events happen on different dates a delta token is added with the prefix "## Delta""
- Max delta is 365 days. If greater than this multiple delta headers are inserted.
- E.g. (1, 1, 2000,...: Event A), (1, 5, 2000, ...: Event B) -> Insert (1, 5, 2000: ## Delta//+4 days)

This input is used for natural language modelling

#### Add quantiles

For structured modelling, quantiles instead of raw values for events such as lab results are required. This is achieved using meds transform's aggregate_code_metadata and bin_numeric_values to bin values into 10% quantiles for each event type

#### Outcomes

We define outcomes using ACES from the output of the ETL pipeline. 7 Operational outcomes are defined as:

| Task | Trigger | Prediction time | Gap / Exclusions | Target window | Label |
|---|---|---|---|---|---|
| LOS - Hospital 3 days | Hospital admission | 24h after admission | If death or discharge between admission and +24h | Hospital admission + 24h -> 3 days | Hospital Discharge |
| LOS - ICU 3 days | ICU admission | 24h after ICU admission | If death or ICU discharge between admission and +24h | ICU admission + 24h -> 3 days | ICU Discharge |
| Mortality - 30d post-discharge | Hospital discharge | At hospital discharge | If death before discharge | Discharge -> +30 days | Death |
| Mortality - 365d post-discharge | Hospital discharge | At hospital discharge | If death before discharge | Discharge -> +365 days | Death |
| Mortality - In-hospital | Hospital admission | 24h after admission | If death or discharge within first 24h | 24h after admission -> until hospital discharge | Death |
| Mortality - In-ICU | ICU admission | 24h after ICU admission | If death or ICU discharge within first 24h | 24h after ICU admission -> until ICU discharge | Death |
| Readmission - 30d post-discharge | Hospital discharge | At hospital discharge | If death before discharge | Discharge -> +30 days | Hospital Admission |

### Modelling

#### Tabular Modelling

We use XGBoost via the MEDS-Tabs package for efficent pre-processing and sweeping of hyperparameters

The sweep is per task and across the following default MEDS-tab parameters using optuna
- model_launcher.model.eta: tag(log, interval(0.001, 1))
- model_launcher.model.lambda: tag(log, interval(0.001, 1))
- model_launcher.model.alpha: tag(log, interval(0.001, 1))
- model_launcher.model.subsample: interval(0.5, 1)
- model_launcher.model.min_child_weight: interval(1e-2, 100)
- model_launcher.model.max_depth: range(2, 16)
- model_launcher.training_params.num_boost_round: range(100, 1000)
- model_launcher.training_params.early_stopping_rounds: range(1, 10)
- tabularization.min_code_inclusion_count=10"
- tabularization.window_sizes=[1d,30d,365d]
- tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
- scale_pos_weight TBC

The best model on the validation set is reloaded and run on the test set.
- Format adherence is 100%
- As the dataset is processed as a single batch, sample latency is calculated as prediction time/len(dataset)

#### Structured Modelling

**Tokenization**

- Create a raw pretraining dataset by joining codes of same subject_id with "//\n//"
- Custom tokenizer then trained
    - Word level tokenizer with words defined by splitting on "//"
    - Min word frequency imposed (of TBC).
    - If code not in tokenizer vocab "UNKNOWN" token is used (matches unknown placeholder from ETL pipeline)
- Pretraining dataset formed by tokenizing raw dataset to maximum length of TBC with TBC overlap between chunks
- Task specific datasets by joining codes up to and including prediction time

**Pre training**

- Standard next token prediction pre-training target
- Using model architecture TBD scaled to size TBD with max model length of TBD, vocab of TBC (padded to multiple of 8)
- Padding on a per batch level
- Training hyperparameters of ....

**Generative Rollout Inference**

- Sort by length to minimize padding per batch
- Stopping criteria
    - Max number of tokens TBD
    - Task dependent
        - End codes
        - Time delta (to day granularity)
- Generation over TBC independent samples
    - Sampling with temp 0 etc...
- Decoding
    - IDs-> tokens
    - Get last input category header (via regex)
    - Set date and time to prediction time
    - Parse line by line (split by "//\n//")
        - If category header -> update current category
        - If time header -> parse and update timestamp (including unspecified)
            -  Timestamp > end time finished = False
        - If delta header -> parse and update time (including unspecified)
             -  Timestamp > end time finished = False
        - Else -> add event (current_timestamp, header+line)
            - If event starts with end code found and finished == True
- Probability via Monte Carlo approx (num positive predictions/num samples)
- format adherence = % timelines that finish

**Embedding + Logistic Regression Training**

- For each dataset
    - Tokenize and truncate task specific dataset
    - Sort by length to minimize padding per batch
    - Extract embedding of final token
    - Normalize embeddings (L2)
    - Hyperparameter sweep over logistic regression
        - lr.C: tag(log, interval(1e-4, 1e2))
        - lr.class_weight: choice("balanced", "null")
        - Uses best AUROC (TBD) on validation set

**Embedding + Logistic Regression Inference**

- Run inference for each task loading correct LR head
- Record sample latency (batch_time/batch_size)
- Format adherence = 100% by default

#### Natural Language

**Create Prompts**

- Create raw EHR texts by
    - Aggregating by subject_id
    - Appending numeric values (rounded to 2 sf) to codes
    - Replace "//" with " " in codes
    - Join codes with newlines
- Create task specific prompts

```
PROMPT_TEMPLATE = """Analyze the following patient Electronic Health Record (EHR) and predict if the following outcome will occur.

# Outcome to Predict
{task_description}

{ehr_title_header}
{ehr}

{instructions_header}
{instructions}
"""

DIRECT_INSTRUCTIONS = """Based on the patient's electronic health record above, predict if the outcome will occur.

## Constraints
Do not explain your reasoning.
Your response must start with "Answer: " followed by either "Yes" or "No".

## Output Format
Answer: Yes/No
"""

THINKING_INSTRUCTIONS = """Based on the patient's electronic health record above, predict if the outcome will occur.

First, write out your step-by-step clinical reasoning inside <think> and </think> tags.
After you have finished thinking, you must provide your final answer.

## Constraints
Your final answer must start with "Answer: " followed by either "Yes" or "No".
Your entire response, including thinking, must be less than {max_tokens} tokens.

## Output Format
Answer: Yes/No
"""
```
- Truncation of EHR to last TBD tokens to fit into template (breaking on newlines)
    - Max length = max_model_length - max_new_tokens
- Direct answers append start of generation message so only 1 token has to be generated (" Yes" or " No").
- Output format chosen to be token efficent whilst ensuring only 2 valid outputs (no degeneracy due to merging with punctuation etc)

**SFT Training**

- Given a task prompt trained to output " Yes" or " No" token
- LORA finetuning TBD
- Weighting TBC?
- Interleave task datasets with weighting
- Standard supervised finetuning setup
- Eval per task

**Local Inference**

- Input prompt with generation config as guided by Qwen documentation
- Sort by length to minimize padding per batch
- Max output tokens = 1 for direct and TBD for thinking
- Probability calculated as softmax of " Yes" and " No" logit
- Format adherence is true if final output token is " Yes" or " No"
- Timing per batch

**Frontier Inference (WIP)**

Need a frontier model which outputs logits and conforms to [MIMIC usage policy](https://physionet.org/news/post/llm-responsible-use/)

To ensure zero data retention whilst using the Gemini models hosted by Google, the following steps have been taken in accordance with this [documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/vertex-ai-zero-data-retention):
- Use Enterprise Vertex offering which guarantees data is not used for training
- Exemption for Abuse Monitoring has been granted
- In memory data caching disabled
- Do not use grounding with search, maps or the live API

Implementation WIP

## Evaluation

For each model we take
- boolean value
- probability
- format adherence
- sample latency

If format adherence is 0 then probability is set to default value of 0.5

And calc the following metrics. These results are all on demo data locally so placeholders more than anything else, and gemini results are missing currently.

### Format Adherence



### Latency

![latency_graph](notebooks/results/outputs/latency.png)


### Cost

TODO
Compute cost x inference time


### AUC ROC


### AUC PRC

### ECE

### Brier Score

## Discussion Points
- Long context window -> truncation is suboptimal
    - Summarisation/selection -> agentic approaches
    - Tabular approaches do not have this issue
- Other natural language presentation formats e.g. json, xml
- Markers of time
    - Chosen due to constraints placed by MIMIC random shift
    - Raw time means something (e.g. year 2150)
    - Days +- from prediction point does not work for general pretraining
    - Using age is coarse estimate as only have year not date of birth
    - Use delta between days. As time of day is informative and not shifted keep time of day rather than hour and minute delta
        - Cons
            - Lose day of week and seasonality effects
            - Time of day not guaranteed to move forward during generative rollout
- Natural language
    - Using logits
    - Could output probability as raw number 0-1
        - Tokenized brier loss
        - Simpler integration with frontier models
- Scaling instruction tuning via MEDS/ACES and this framework


## Open Questions

The following are intentionally out of scope but could be made in scope, depending on importance, timelines, capacity. If not done will move to discussion/future work section in write up!
- OMOP instead of MIMIC specific descriptions?
- Temporal split: ~possible but messy due to MIMIC's random shift
- Additional tasks: e.g. labs
    - Beyond binary prediction?
- Recent Nature comment piece (Shah et al.) points out 5 common areas missed in evaluation (could be v nice framing)
    - Event frequency (could add)
    - Assessment of calibration (done)
    - Timeline completion rates (done)
    - Assessment of reliance on shortcuts
        - Not retraining without admin codes
        - But explicit about tokenization
    - External validation
        - Big lift (eICU?)
- GRPO to finetune reasoning model

General unknowns
- Context window
    - How long?
    - How to make fair? context_window  = max_model_length - max_new_tokens
- Using "all the data we have"
    - Lots of ICU observations
    - Not using MIMIC-ED
-


## TODOs

Coding
- Config
    - Inheritance
    - Structure
- QOL
    - Docstrings
    - Logs consitent
- Flash attention
- Preprocessing numbers
    - Token length types etc...
    - Generative rollout -> ground truth how tokens to complete
- Weighted losses?
- Gemini implementation
- All eval graphs

Move to HPC
- Confirm full run configs
- Run mimic demo through HPC
- Run "full" mimic through HPC
