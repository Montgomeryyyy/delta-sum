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
- Post hoc calibration

General unknowns
- Context window
    - How long?
    - How to make fair? context_window  = max_model_length - max_new_tokens
- Using "all the data we have"
    - Lots of ICU observations
    - Not using MIMIC-ED

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
- Add calibration, ROC and PRC graphs per task

Move to HPC
- Confirm full run configs
- Run mimic demo through HPC
- Run "full" mimic through HPC

## Discussion
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


## Methods

The following is the current methods in excruciating detail, was mainly an exercise for me to write out what is happening and check this is true... (and start of paper's method section)

No one should feel obliged to read this! But in case of interest, this code is implemented and running on MIMIC-demo on my laptop currently.

Still some experimental decisions for the full data run are flagged by TBC/TBD (e.g. model sizes etc...) but the data flow is all there.

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

- Split into train/tuning/held_out (80%/10%/10%) on subject_ids
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
- Direct answers append start of generation message and "Answer: " so only 1 token has to be generated (" Yes" or " No").
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

And calc the following metrics.

These results are all on demo data locally so placeholders more than anything else, and gemini results are missing currently.

TODO: Add calibration, ROC and PR graphs per task

### Format Adherence


| data             | method             |   format_adherence |   ci_lower |   ci_upper |
|:-----------------|:-------------------|-------------------:|-----------:|-----------:|
| Tabular          | MEDS-Tab           |           1        |  1         |   1        |
| Structured       | Generative Rollout |           0        |  0         |   0        |
| Structured       | Embedding + LR     |           1        |  1         |   1        |
| Natural Language | Qwen 0.6B Thinking |           0.150943 |  0.0849057 |   0.216981 |
| Natural Language | Qwen 0.6B Direct   |           1        |  1         |   1        |
| Natural Language | Qwen 0.6B SFT      |           1        |  1         |   1        |

*Note that due to v low max new tokens generative rollout and thinking are failing to complete.

### Latency

![latency_graph](notebooks/results/outputs/latency.png)


| data             | method             |     median |    ci_lower |    ci_upper |
|:-----------------|:-------------------|-----------:|------------:|------------:|
| Tabular          | MEDS-Tab           | 0.00025505 | 0.000231225 | 0.000266914 |
| Structured       | Embedding + LR     | 0.128581   | 0.128581    | 0.163542    |
| Structured       | Generative Rollout | 2.47751    | 2.46002     | 2.53897     |
| Natural Language | Qwen 0.6B Direct   | 0.0470985  | 0.0470826   | 0.0473876   |
| Natural Language | Qwen 0.6B SFT      | 0.0499768  | 0.0498709   | 0.0500249   |
| Natural Language | Qwen 0.6B Thinking | 1.96869    | 1.92932     | 2.09495     |

### Cost

TODO
Compute cost x inference time


### AUC ROC

| data             | method             | Los Hosp 3 Days    | Los Icu 3 Days       | Mortality Hosp Discharge 30 Days   | Mortality Hosp Discharge 365 Days   | Mortality In Hosp    | Mortality In Icu     |   Readmission Hosp 30 Days | MACRO                |
|:-----------------|:-------------------|:-------------------|:---------------------|:-----------------------------------|:------------------------------------|:---------------------|:---------------------|---------------------------:|:---------------------|
| Tabular          | MEDS-Tab           | 0.5 [0.5, 0.5]     | 0.5 [0.5, 0.5]       | 0.5 [0.5, 0.5]                     | 0.5 [0.5, 0.5]                      | 0.5 [0.5, 0.5]       | 0.5 [0.5, 0.5]       |                        nan | 0.5 [0.5, 0.5]       |
| Structured       | Embedding + LR     |                    | 0.375 [0.0, 0.75]    | 1.0 [1.0, 1.0]                     | 0.667 [0.071, 1.0]                  | 0.0 [0.0, 0.0]       | 1.0 [1.0, 1.0]       |                            | 0.608 [0.267, 0.853] |
| Structured       | Generative Rollout | 0.5 [0.5, 0.5]     | 0.5 [0.5, 0.5]       | 0.5 [0.5, 0.5]                     | 0.5 [0.5, 0.5]                      |                      | 0.5 [0.5, 0.5]       |                        nan | 0.5 [0.5, 0.5]       |
| Natural Language | Qwen 0.6B Direct   | 0.182 [0.0, 0.465] | 0.3 [0.022, 0.625]   | 0.667 [0.4, 0.929]                 | 0.462 [0.214, 0.733]                | 0.214 [0.071, 0.429] | 0.462 [0.167, 0.769] |                        nan | 0.381 [0.221, 0.523] |
| Natural Language | Qwen 0.6B SFT      | 0.182 [0.0, 0.455] | 0.325 [0.061, 0.651] | 0.667 [0.4, 0.867]                 | 0.487 [0.218, 0.75]                 | 0.286 [0.071, 0.548] | 0.462 [0.231, 0.75]  |                        nan | 0.401 [0.24, 0.521]  |
| Natural Language | Qwen 0.6B Thinking | 0.341 [0.0, 0.786] | 0.4 [0.0, 0.909]     | 0.6 [0.339, 0.825]                 | 0.949 [0.786, 1.0]                  | 1.0 [1.0, 1.0]       | 0.615 [0.333, 0.846] |                        nan | 0.651 [0.448, 0.787] |

![macro_roc_auc](notebooks/results/outputs/macro_roc_auc.png)


### AUC PRC

| data             | method             | Los Hosp 3 Days      | Los Icu 3 Days       | Mortality Hosp Discharge 30 Days   | Mortality Hosp Discharge 365 Days   | Mortality In Hosp    | Mortality In Icu     |   Readmission Hosp 30 Days | MACRO                |
|:-----------------|:-------------------|:---------------------|:---------------------|:-----------------------------------|:------------------------------------|:---------------------|:---------------------|---------------------------:|:---------------------|
| Tabular          | MEDS-Tab           | 0.267 [0.067, 0.533] | 0.714 [0.429, 0.929] | 0.063 [0.063, 0.188]               | 0.188 [0.063, 0.375]                | 0.067 [0.067, 0.2]   | 0.071 [0.071, 0.214] |                        nan | 0.228 [0.188, 0.408] |
| Structured       | Embedding + LR     |                      | 0.7 [0.447, 0.961]   | 1.0 [1.0, 1.0]                     | 0.571 [0.071, 1.0]                  | 0.067 [0.067, 0.2]   | 1.0 [1.0, 1.0]       |                            | 0.668 [0.403, 0.92]  |
| Structured       | Generative Rollout | 0.267 [0.067, 0.49]  | 0.714 [0.429, 0.929] | 0.063 [0.063, 0.25]                | 0.188 [0.063, 0.375]                |                      | 0.071 [0.071, 0.245] |                        nan | 0.26 [0.203, 0.455]  |
| Natural Language | Qwen 0.6B Direct   | 0.207 [0.083, 0.458] | 0.684 [0.429, 0.943] | 0.167 [0.108, 0.635]               | 0.216 [0.091, 0.518]                | 0.083 [0.071, 0.308] | 0.125 [0.091, 0.443] |                        nan | 0.247 [0.221, 0.474] |
| Natural Language | Qwen 0.6B SFT      | 0.207 [0.077, 0.442] | 0.717 [0.453, 0.946] | 0.167 [0.111, 0.6]                 | 0.222 [0.091, 0.533]                | 0.091 [0.071, 0.333] | 0.125 [0.091, 0.454] |                        nan | 0.255 [0.232, 0.471] |
| Natural Language | Qwen 0.6B Thinking | 0.268 [0.091, 0.667] | 0.655 [0.431, 0.988] | 0.143 [0.1, 0.5]                   | 0.867 [0.333, 1.0]                  | 1.0 [1.0, 1.0]       | 0.167 [0.111, 0.571] |                        nan | 0.517 [0.399, 0.762] |

![macro_auprc](notebooks/results/outputs/macro_auprc.png)


### Brier Score

| data             | method             | Los Hosp 3 Days      | Los Icu 3 Days       | Mortality Hosp Discharge 30 Days   | Mortality Hosp Discharge 365 Days   | Mortality In Hosp    | Mortality In Icu     | Readmission Hosp 30 Days   | MACRO                |
|:-----------------|:-------------------|:---------------------|:---------------------|:-----------------------------------|:------------------------------------|:---------------------|:---------------------|:---------------------------|:---------------------|
| Tabular          | MEDS-Tab           | 0.2 [0.132, 0.29]    | 0.214 [0.163, 0.28]  | 0.059 [0.002, 0.172]               | 0.154 [0.052, 0.256]                | 0.063 [0.002, 0.185] | 0.067 [0.009, 0.182] | 0.046 [0.046, 0.046]       | 0.114 [0.084, 0.151] |
| Structured       | Embedding + LR     |                      | 0.263 [0.245, 0.285] | 0.25 [0.249, 0.251]                | 0.246 [0.216, 0.278]                | 0.249 [0.246, 0.253] | 0.21 [0.164, 0.252]  |                            | 0.244 [0.231, 0.255] |
| Structured       | Generative Rollout | 0.267 [0.067, 0.468] | 0.714 [0.429, 0.929] | 0.063 [0.0, 0.188]                 | 0.188 [0.0, 0.375]                  |                      | 0.071 [0.0, 0.214]   | 0.0 [0.0, 0.0]             | 0.217 [0.144, 0.294] |
| Natural Language | Qwen 0.6B Direct   | 0.455 [0.332, 0.575] | 0.252 [0.148, 0.366] | 0.093 [0.052, 0.164]               | 0.162 [0.063, 0.267]                | 0.091 [0.033, 0.188] | 0.091 [0.046, 0.167] | 0.3 [0.272, 0.328]         | 0.206 [0.173, 0.242] |
| Natural Language | Qwen 0.6B SFT      | 0.304 [0.213, 0.39]  | 0.355 [0.266, 0.453] | 0.059 [0.007, 0.162]               | 0.165 [0.008, 0.322]                | 0.065 [0.004, 0.185] | 0.067 [0.006, 0.187] | 0.062 [0.053, 0.072]       | 0.154 [0.115, 0.196] |
| Natural Language | Qwen 0.6B Thinking | 0.499 [0.308, 0.711] | 0.462 [0.232, 0.696] | 0.403 [0.226, 0.596]               | 0.141 [0.023, 0.296]                | 0.029 [0.007, 0.057] | 0.332 [0.158, 0.527] | 0.345 [0.156, 0.564]       | 0.316 [0.246, 0.387] |

![macro_brier](notebooks/results/outputs/macro_brier.png)

### ECE

| data             | method             | Los Hosp 3 Days      | Los Icu 3 Days       | Mortality Hosp Discharge 30 Days   | Mortality Hosp Discharge 365 Days   | Mortality In Hosp    | Mortality In Icu     | Readmission Hosp 30 Days   | MACRO                |
|:-----------------|:-------------------|:---------------------|:---------------------|:-----------------------------------|:------------------------------------|:---------------------|:---------------------|:---------------------------|:---------------------|
| Tabular          | MEDS-Tab           | 0.421 [0.353, 0.511] | 0.45 [0.4, 0.517]    | 0.105 [0.048, 0.218]               | 0.329 [0.227, 0.432]                | 0.103 [0.042, 0.225] | 0.153 [0.096, 0.269] | 0.214 [0.214, 0.214]       | 0.254 [0.224, 0.29]  |
| Structured       | Embedding + LR     |                      | 0.714 [0.5, 0.929]   | 0.063 [0.0, 0.188]                 | 0.188 [0.063, 0.375]                | 0.067 [0.0, 0.2]     | 0.071 [0.0, 0.214]   |                            | 0.22 [0.145, 0.296]  |
| Structured       | Generative Rollout | 0.267 [0.067, 0.468] | 0.714 [0.429, 0.929] | 0.063 [0.0, 0.188]                 | 0.188 [0.0, 0.375]                  |                      | 0.071 [0.0, 0.214]   | 0.0 [0.0, 0.0]             | 0.217 [0.144, 0.294] |
| Natural Language | Qwen 0.6B Direct   | 0.267 [0.067, 0.533] | 0.714 [0.429, 0.929] | 0.063 [0.0, 0.188]                 | 0.188 [0.0, 0.375]                  | 0.067 [0.0, 0.2]     | 0.071 [0.0, 0.214]   | 0.0 [0.0, 0.0]             | 0.196 [0.13, 0.261]  |
| Natural Language | Qwen 0.6B SFT      | 0.267 [0.067, 0.467] | 0.714 [0.429, 0.929] | 0.063 [0.0, 0.188]                 | 0.188 [0.0, 0.375]                  | 0.067 [0.0, 0.2]     | 0.071 [0.0, 0.214]   | 0.0 [0.0, 0.0]             | 0.196 [0.13, 0.261]  |
| Natural Language | Qwen 0.6B Thinking | 0.267 [0.067, 0.533] | 0.714 [0.5, 0.929]   | 0.063 [0.0, 0.188]                 | 0.188 [0.0, 0.375]                  | 0.067 [0.0, 0.2]     | 0.071 [0.0, 0.214]   | 0.0 [0.0, 0.0]             | 0.196 [0.135, 0.261] |

![macro_ece](notebooks/results/outputs/macro_ece.png)
