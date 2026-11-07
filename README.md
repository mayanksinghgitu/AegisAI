# AegisAI -- Adaptive Multimodal AI for Personalized Healthcare & Wellness Risk Assessment

**Project ID:** SOAI-033\
**Project Type:** Major Project \| B.Tech CSE (AI--ML)\
**Institution:** Galgotias University, Greater Noida, India\
**Academic Year:** 2026--2027

AegisAI is a student research prototype for **non-diagnostic
wellness-support risk assessment**. It is designed to bring together
heterogeneous wellness signals and estimate a three-level support
priority: **Low, Moderate, or High**. The project combines a proposed
quality-aware multimodal architecture with an implemented
structured-data machine-learning pipeline and desktop training
environment called the **AegisAI Advanced Control Center**.

> **Important scope note:** The reported experiments evaluate the
> structured-feature estimator. The complete raw-image/raw-audio
> encoders, learned modality-quality scorers, masked adaptive fusion
> gate, calibration, SHAP explanations, and recommendation/safety layers
> are described as parts of the proposed framework, but are not yet
> trained and empirically validated as a complete multimodal pipeline.
> The reported model metrics must not be interpreted as clinical
> accuracy or evidence of medical benefit.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Problem Statement](#problem-statement)
-   [Objectives](#objectives)
-   [Proposed Multimodal Inputs](#proposed-multimodal-inputs)
-   [System Architecture](#system-architecture)
-   [Implemented Components](#implemented-components)
-   [Machine Learning Models](#machine-learning-models)
-   [Dataset](#dataset)
-   [Experimental Methodology](#experimental-methodology)
-   [Results](#results)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Installation and Execution](#installation-and-execution)
-   [Limitations and Future Work](#limitations-and-future-work)
-   [Ethics, Privacy, and Intended
    Use](#ethics-privacy-and-intended-use)
-   [Project Team](#project-team)

## Project Overview

Personal wellness is influenced by multiple interacting factors,
including sleep, physical activity, diet, hydration, stress, symptoms,
emotional indicators, physiological measurements, and environmental
conditions. Considering only one signal can provide an incomplete
picture.

AegisAI proposes a unified framework that can use available information,
account for missing or low-quality modalities, and provide a cautious
wellness-support priority with understandable contributing factors.
Users may skip optional camera or microphone inputs.

The current working implementation focuses on structured-data
classification and model experimentation through the AegisAI Advanced
Control Center.

## Problem Statement

Develop an AI-based wellness assistant that can integrate
facial-expression indicators, voice characteristics, physiological
measurements, user-reported symptoms, lifestyle and dietary information,
and environmental factors to estimate general wellness-support priority
and provide explainable wellness suggestions---without replacing
professional medical diagnosis or treatment.

## Objectives

1.  Design a unified architecture for affective, physiological,
    behavioural, self-reported, and environmental inputs.
2.  Specify a quality-aware adaptive fusion mechanism that accounts for
    modality availability and quality.
3.  Implement a leakage-safe machine-learning pipeline for three-class
    wellness-support priority estimation from structured features.
4.  Train and compare linear, bagging, boosting, and neural model
    families.
5.  Select a production candidate using validation Macro F1 and an
    overfitting-gap diagnostic.
6.  Build a control center for model training, monitoring, comparison,
    evaluation, and artifact saving.
7.  Define conservative handling for uncertainty, missing data, privacy,
    and professional-support escalation.

## Proposed Multimodal Inputs

  -----------------------------------------------------------------------
  Signal group                        Example information
  ----------------------------------- -----------------------------------
  Facial                              Expression-related features derived
                                      from a camera frame or short video,
                                      with user consent

  Voice                               Acoustic descriptors or embeddings
                                      derived from a short recording,
                                      with user consent

  Physiological                       Heart rate, blood pressure, oxygen
                                      saturation, body temperature,
                                      respiratory rate

  Self-reported                       Symptoms, mood, perceived stress,
                                      and fatigue

  Lifestyle and dietary               Sleep, physical activity, diet, and
                                      water intake

  Environmental                       Temperature, humidity, and
                                      air-quality indicators
  -----------------------------------------------------------------------

The proposed system associates inputs with timestamps, units,
availability, and quality information. Missing modalities are intended
to be masked rather than replaced with fabricated sensor readings.

## System Architecture

The proposed full framework follows this high-level workflow:

1.  **Input collection:** Receive any available subset of the six signal
    groups.
2.  **Validation and quality assessment:** Check units, ranges,
    timestamps, plausibility, and signal quality.
3.  **Modality-specific encoding:** Convert each available modality into
    a representation.
4.  **Quality-aware adaptive fusion:** Use availability masks and
    modality-quality scores to weight usable evidence.
5.  **Wellness-support estimation:** Estimate probabilities for Low,
    Moderate, and High priority.
6.  **Uncertainty and safety checks:** Consider confidence,
    out-of-distribution inputs, missing information, and safety rules.
7.  **Output:** Return an explained estimate, an inconclusive result, or
    a professional-support message, as appropriate.

This describes the **proposed architecture**. The currently evaluated
implementation is the structured-data model-training and evaluation
pipeline described below.

## Implemented Components

The AegisAI Advanced Control Center is a desktop training environment
described in the project report. Its documented capabilities include:

-   Loading the structured CSV dataset.
-   Stratified sampling when the source dataset exceeds 300,000 rows.
-   Stratified 70/15/15 train/validation/test splitting.
-   Numerical and categorical preprocessing fitted on training data
    only.
-   Individual model training and a **Train All Models** workflow.
-   Live, time-stamped training logs.
-   Model comparison using training and validation metrics.
-   Production-model selection using validation Macro F1.
-   Overfitting-gap reporting.
-   One-time evaluation on the held-out test set after model selection.
-   Saving the selected model, metrics CSV, and text evaluation report.

The report identifies these output artifacts:

-   `AegisAI_Health_Risk_Model.joblib`
-   `AegisAI_Model_Metrics.csv`
-   `AegisAI_Model_Evaluation_Report.txt`

## Machine Learning Models

Eight models were compared:

  Model                             Model family
  --------------------------------- --------------------------
  Logistic Regression               Linear
  Random Forest                     Bagging / tree ensemble
  Extra Trees                       Randomized tree ensemble
  Artificial Neural Network (ANN)   Neural network
  XGBoost                           Gradient boosting
  HistGradientBoosting              Gradient boosting
  LightGBM                          Gradient boosting
  CatBoost                          Gradient boosting

**Model-selection criterion:** Highest validation Macro F1, considered
together with the train-to-validation Macro F1 gap. The test set is kept
untouched until after model selection.

## Dataset

The project report documents the following dataset summary:

  -------------------------------------------------------------------------------------------------
  Property                                                                                    Value
  ------------------------------ ------------------------------------------------------------------
  Dataset file                     `AegisAI_Multimodal_Health_Wellness_Risk_Dataset_400K_Final.csv`

  Original size                                                           400,000 rows × 42 columns

  Missing cells                                          414,245 (approximately 2.47% of all cells)

  Sample used                                                                          300,000 rows

  Input features                                                   39 (31 numerical, 8 categorical)

  Processed features                                                                             70

  Target classes                                                                High, Low, Moderate

  Sample class distribution                                               100,000 records per class
  -------------------------------------------------------------------------------------------------

The 300,000-row sample is split as follows:

  Partition      Records   Percentage
  ------------ --------- ------------
  Training       210,000          70%
  Validation      45,000          15%
  Test            45,000          15%

Each partition is stratified by class. The test set contains 15,000
records per class.

**Dataset availability and provenance:** The report names the dataset
file but does not establish a public download location or fully document
whether the data are real-world or synthetic. Add the dataset only if
you have the right to share it, and document its provenance and license
before publishing it. Do not assume that the dataset is included in this
repository.

## Experimental Methodology

-   A stratified sample of 300,000 records is used from the 400,000-row
    dataset.
-   The data are divided into training, validation, and test partitions
    using a 70/15/15 split.
-   Numerical and categorical preprocessing is fitted only on the
    training partition, then applied unchanged to validation and test
    data.
-   Model selection is based on validation Macro F1.
-   The train-to-validation Macro F1 gap is tracked as an overfitting
    diagnostic; the preferred range in the report is 0--2%.
-   The held-out test set is evaluated only after model selection.
-   Accuracy, balanced accuracy, Macro F1, Weighted F1, per-class
    metrics, and a confusion matrix are reported.

## Results

**LightGBM was selected as the production model** in the reported
experiment.

  Metric                     LightGBM result
  ------------------------ -----------------
  Test accuracy                   **90.30%**
  Test balanced accuracy              90.30%
  Test Macro F1                   **0.9031**
  Test Weighted F1                    0.9031
  Validation Macro F1                 0.9006
  Training Macro F1                   0.9197
  Train--validation gap            **1.92%**
  Training time                68.46 seconds

### Test Macro F1 -- Model Comparison

  Model                    Test Macro F1   Test Accuracy
  ---------------------- --------------- ---------------
  LightGBM                    **0.9031**      **90.30%**
  HistGradientBoosting            0.8871          88.68%
  XGBoost                         0.8823          88.19%
  CatBoost                        0.8786          87.82%
  ANN                             0.8560          85.58%
  Logistic Regression             0.8366          83.61%
  Random Forest                   0.8036          80.47%
  Extra Trees                     0.7855          78.99%

### LightGBM -- Per-Class F1

  Class        F1-score
  ---------- ----------
  High           0.9305
  Low            0.9225
  Moderate       0.8564

The Moderate class is the weakest-performing class in the reported test
results. The report also notes that most classification errors involve
the Moderate class.

> These results describe the model's performance on the documented
> structured dataset and split only. They do not establish real-world,
> clinical, or diagnostic performance.

## Technology Stack

The report documents the following tools and libraries:

-   **Language:** Python 3
-   **Data processing:** pandas, NumPy
-   **Machine learning and preprocessing:** scikit-learn
-   **Gradient boosting:** XGBoost, LightGBM, CatBoost,
    HistGradientBoosting
-   **Model persistence:** joblib
-   **Neural baseline:** neural-network module (the report does not
    specify its exact package)
-   **Analysis and plots:** Matplotlib
-   **Report preparation:** LaTeX / Overleaf

The report does not specify exact package versions or a complete
dependency lock file. Record the versions used by your actual project
environment before publishing a reproducible setup.

## Project Structure

The exact repository tree and application entry-point filename are not
specified in the report. A possible structure for documenting the
repository is:

``` text
AegisAI/
├── README.md
├── data/
│   └── dataset.csv                 # if permitted to distribute
├── models/
│   └── AegisAI_Health_Risk_Model.joblib
├── outputs/
│   ├── AegisAI_Model_Metrics.csv
│   └── AegisAI_Model_Evaluation_Report.txt
├── notebooks/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   └── control_center/
└── requirements.txt
```

This is an illustrative organization, not a claim about the current
repository contents. Update it to match the actual files and folders in
your GitHub repository.

## Installation and Execution

The project report does not provide a verified application entry-point
filename, exact GUI framework, or pinned dependency versions. The
following is a starting point; adapt it to the actual source files and
environment.

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a tested `requirements.txt`:

``` bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by the actual implementation.
The report documents pandas, NumPy, scikit-learn, XGBoost, LightGBM,
CatBoost, and joblib; the exact neural-network and GUI dependencies must
be confirmed from the source code.

### 4. Add the dataset

Place the permitted dataset file at the path expected by the
application:

``` text
AegisAI_Multimodal_Health_Wellness_Risk_Dataset_400K_Final.csv
```

The dataset is not assumed to be publicly available or bundled with this
repository.

### 5. Launch the Control Center

Run the actual Control Center entry-point script from the repository.
Replace the placeholder below with the real filename after checking the
source code:

``` bash
python <control_center_entry_point>.py
```

From the desktop interface, use **Train All Models** or the individual
model controls, if available in your checked-out implementation.

## Limitations and Future Work

The project report identifies the following work as pending or requiring
further validation:

-   Train and validate the raw-image and raw-audio modality branches.
-   Implement and evaluate learned modality-quality scorers and the
    masked adaptive fusion gate.
-   Validate the target using a recognised wellness instrument and a
    documented labelling protocol.
-   Collect properly aligned multimodal data and use participant-grouped
    splits where participant identifiers exist.
-   Add probability calibration, Brier score, Expected Calibration Error
    (ECE), and calibrated abstention.
-   Evaluate missing-modality patterns, noisy inputs, and
    out-of-distribution cases.
-   Conduct subgroup fairness analysis and report uncertainty intervals.
-   Integrate and review SHAP explanations and the bounded
    recommendation library.
-   Validate safety and escalation rules with qualified health
    professionals.
-   Run repeated seeds and external or temporal evaluation.

## Ethics, Privacy, and Intended Use

AegisAI is an educational research prototype for general wellness
support. It is **not a medical diagnostic system** and must not be used
as a substitute for professional medical advice, diagnosis, or
treatment.

-   Facial and voice inputs should remain optional and require explicit
    user consent.
-   Do not infer a mental-health diagnosis from facial expressions or
    vocal characteristics.
-   Avoid storing raw audio or video when derived features are
    sufficient.
-   A deployed system should use appropriate encryption, access control,
    retention limits, deletion support, and audit logs.
-   Low-confidence, invalid, or insufficient inputs should lead to
    cautious or inconclusive handling rather than false certainty.
-   Any urgent-symptom escalation workflow requires review and approval
    by qualified health professionals.

## Project Team

  Name           Role
  -------------- ---------------------
  Mayank Singh   Project team member
  Mayank         Project team member

**Project Supervisor:** Ms. Jyoti Ratna\
Assistant Professor, Department of AI & Data Science\
Galgotias University, Greater Noida, Uttar Pradesh, India

------------------------------------------------------------------------

**Project ID:** SOAI-033\
**Academic Year:** 2026--2027
