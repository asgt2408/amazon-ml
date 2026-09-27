# Business Entity Resolution — ML Challenge 2026

## 1. Project Overview

This project implements a scalable machine-learning pipeline for business entity resolution across three independent data sources.

The task is to identify, for every Source 1 business entity, all corresponding records in Source 2 and Source 3. A Source 1 entity may have:

* no matching records,
* one matching record, or
* multiple matching records.

The solution is designed around two major stages:

1. **Blocking / candidate generation** to reduce the search space.
2. **Supervised pair matching** to classify candidate pairs as matches or non-matches.

The final system uses DuckDB for large-scale candidate generation and feature processing and scikit-learn's `HistGradientBoostingClassifier` for final pair scoring.

---

## 2. Repository Structure

```text
student_resource/
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── normalize.py
│   ├── blocking.py
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│   └── test_blocking.py
│
├── utils/
│   └── validate_submission.py
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── Documentation_template.md
└── requirements.txt
```

Generated intermediate files such as DuckDB databases, Parquet feature files, and trained model files are not required as source inputs when reproducing the pipeline; they can be generated during execution.

---

# 3. Problem Definition

Each source contains:

```text
entity_id
business_name
business_address
country
```

Source 1 is the reference source.

For every Source 1 entity, the system predicts the matching entity IDs from Source 2 and Source 3.

The output contains:

```text
source1_entity_id    matched_entity_ids
```

where `matched_entity_ids` is a comma-separated list.

Entities for which no reliable match is found receive an empty list.

---

# 4. Design Goals

The implementation was designed with the following goals:

* Avoid all-pairs comparison between millions of records.
* Preserve candidate recall while keeping candidate sets small.
* Handle noisy business names.
* Handle noisy and partially missing addresses.
* Preserve multilingual Unicode text during normalization.
* Treat country as an open-set attribute.
* Use precision-aware matching because the challenge evaluates macro F0.5.
* Avoid external business databases, APIs, geocoding services, or external identity lookup.
* Process millions of records using DuckDB rather than loading every pair into Python memory.

---

# 5. Pipeline

```text
Raw TSV files
     |
     v
Unicode-safe normalization
     |
     v
Blocking / candidate generation
     |
     v
Candidate pairs
     |
     v
Pair feature generation
     |
     v
HistGradientBoostingClassifier
     |
     v
Probability threshold = 0.70
     |
     v
Final matched IDs
     |
     +----------------------+
     |                      |
     v                      v
matching_results.tsv    candidate_pairs.tsv
```

---

# 6. Normalization

Normalization is implemented in `src/normalize.py`.

The normalizer is Unicode-aware rather than converting all text to ASCII. This is important because the test data contains multiple languages.

## General text normalization

The normalization process includes:

* Unicode NFKC normalization.
* Lowercasing.
* Punctuation-to-space conversion.
* Preservation of Unicode alphanumeric characters.
* Whitespace normalization.
* Empty/null handling.

This avoids destroying non-Latin scripts during preprocessing.

## Business-name normalization

Business names are normalized using domain-specific legal/business abbreviations.

The pipeline also creates a core representation that removes common legal-form terms so that variations such as legal suffixes do not dominate similarity.

Examples of normalization targets include variations of:

```text
corporation
corp
company
co
limited
ltd
private
pvt
incorporated
inc
```

The exact mappings are implemented in `normalize.py`.

## Address normalization

Address normalization handles common address variation patterns such as:

* road/street abbreviations,
* apartment/unit terminology,
* common address abbreviations,
* punctuation,
* numeric components,
* whitespace variation.

Address numbers are extracted separately because they provide useful blocking and matching information.

## Country

Country is normalized without restricting the pipeline to a predefined list.

This is important because the challenge explicitly treats country as an open-set field.

---

# 7. Blocking / Candidate Generation

Blocking is implemented in `src/blocking.py`.

A naive all-pairs comparison would require comparing millions of Source 1 records against millions of Source 2/3 records and is therefore infeasible.

Instead, the system creates several blocking keys.

## Blocking key 1 — Country + exact normalized name

```text
country + normalized_full_name
```

This captures high-confidence exact name matches while limiting collisions by country.

## Blocking key 2 — Country + first/last name tokens

```text
country + first_token + last_token
```

This allows matches where minor internal name differences exist.

## Blocking key 3 — Country + first two name tokens

```text
country + first_token + second_token
```

This provides another path for name variations.

## Blocking key 4 — Country + address number + first address token

```text
country + address_number + first_address_token
```

This combines a strong numeric address signal with a location token.

## Blocking key 5 — Rare name token

Rare name tokens are indexed separately.

Only tokens satisfying frequency constraints are used, preventing common terms from generating huge candidate blocks.

The implementation uses a bounded frequency range so that extremely common words do not create excessive candidate sets.

## Blocking key 6 — Rare address token

Rare address tokens provide another independent path for records whose business names differ significantly.

---

# 8. Block Size Control

Large blocks are restricted to avoid exploding the candidate set.

The implemented blocker uses a maximum block size of approximately 100 records for its indexed blocking groups.

This prevents very common tokens from creating millions of candidate pairs.

---

# 9. Full Test Blocking Results

The final test blocking run processed:

```text
Source 1 : 1,732,544
Source 2 : 4,887,273
Source 3 : 5,082,316
```

The resulting candidate set contained:

```text
140,995,004 candidate pairs
```

Average candidates per Source 1 entity:

```text
~81.36
```

The blocker produced zero candidates for:

```text
3,763 Source 1 entities
```

Those entities are explicitly retained in `candidate_pairs.tsv` with an empty candidate list, as required by the submission validator.

---

# 10. Training Data Construction

Training candidate pairs were generated using the same blocking philosophy.

The ground-truth file was used to label candidate pairs.

Each pair receives:

```text
label = 1
```

when the candidate entity is a known match, otherwise:

```text
label = 0
```

The resulting feature dataset contained:

```text
24,093,709 candidate pairs
```

with:

```text
6,351,170 positive pairs
17,742,539 negative pairs
```

---

# 11. Feature Engineering

The feature generation stage intentionally uses inexpensive features so that millions of candidate pairs can be processed efficiently.

The final model uses 16 features.

## Country features

### `country_match`

Whether normalized country values are equal.

---

## Name features

### `name_exact`

Exact equality between normalized business names.

### `name_first_match`

Whether the first name token matches.

### `name_last_match`

Whether the last name token matches.

### `name_first_two_match`

Whether the first two name tokens match.

### `name_contains`

Whether one normalized name representation contains the other.

### `name_length_diff`

Absolute difference in normalized name length.

### `name_length_ratio`

Ratio of normalized name lengths.

### `name_missing`

Indicates missing name information.

---

## Address features

### `address_exact`

Exact equality between normalized addresses.

### `address_first_match`

Whether the first normalized address token matches.

### `address_number_match`

Whether extracted address numbers match.

### `address_contains`

Whether one normalized address contains the other.

### `address_length_diff`

Absolute difference in normalized address length.

### `address_length_ratio`

Ratio of normalized address lengths.

### `address_missing`

Indicates missing address information.

---

# 12. Model

The final classifier is:

```text
sklearn.ensemble.HistGradientBoostingClassifier
```

The model was selected because it provides nonlinear decision boundaries while remaining practical for a large tabular feature dataset.

No external language model or external business-identity service is used.

The trained model is stored as:

```text
matching_model.joblib
```

---

# 13. Validation Strategy

The training data was split by Source 1 entity rather than randomly splitting individual pairs.

This prevents candidate pairs belonging to the same Source 1 entity from being distributed between training and validation sets.

The validation partition was selected using:

```text
hash(source1_entity_id) % 5 == 0
```

This provides an entity-level validation split.

---

# 14. Threshold Selection

The model produces a match probability.

Because the challenge evaluates macro F0.5 and places greater emphasis on precision, multiple thresholds were evaluated.

The validation results included:

| Threshold |  Precision |     Recall |       F0.5 |
| --------: | ---------: | ---------: | ---------: |
|      0.50 |     0.9229 |     0.7536 |     0.8832 |
|      0.55 |     0.9376 |     0.7370 |     0.8892 |
|      0.60 |     0.9458 |     0.7271 |     0.8921 |
|      0.65 |     0.9535 |     0.7153 |     0.8939 |
|  **0.70** | **0.9715** | **0.6842** | **0.8962** |
|      0.75 |     0.9787 |     0.6682 |     0.8955 |
|      0.80 |     0.9840 |     0.6536 |     0.8936 |
|      0.85 |     0.9871 |     0.6413 |     0.8910 |
|      0.90 |     0.9913 |     0.6169 |     0.8840 |

The selected threshold was:

```text
0.70
```

because it produced the highest measured validation F0.5 among the evaluated thresholds.

---

# 15. Test Prediction

The final prediction process:

1. Load test Source 1, Source 2 and Source 3.
2. Normalize records.
3. Build the same blocking indexes.
4. Generate candidate pairs.
5. Generate the same 16 model features.
6. Load the trained classifier.
7. Generate match probabilities.
8. Keep candidate pairs with:

```text
probability >= 0.70
```

9. Group predictions by Source 1 entity.
10. Write one output row per Source 1 entity.

---

# 16. Final Test Results

The final prediction run produced:

```text
Test Source 1 entities : 1,732,544
Candidate pairs        : 140,995,004
Predicted candidate rows: 4,590,539
S1 entities with match : 1,511,135
S1 entities empty      : 221,409
Threshold              : 0.70
```

Final matching output:

```text
output/matching_results.tsv
```

Final candidate output:

```text
output/candidate_pairs.tsv
```

---

# 17. Output Format

## matching_results.tsv

```text
source1_entity_id    matched_entity_ids
```

Example:

```text
S1-00001    S2-00047,S3-00812
S1-00002    S3-00004
S1-00003
```

Every Source 1 entity has exactly one row.

An empty second field represents no predicted match.

## candidate_pairs.tsv

```text
source1_entity_id    candidate_entity_ids
```

Example:

```text
S1-00001    S2-00047,S2-00193,S3-00812
S1-00002    S3-00004
S1-00003
```

Every Source 1 entity has exactly one row.

---

# 18. Validation

The challenge-provided validator is:

```text
utils/validate_submission.py
```

Run:

```bash
python3 utils/validate_submission.py
```

The final generated files passed the validator with:

```text
matching_results.tsv: 1,732,544 rows
candidate_pairs.tsv: 1,732,544 rows
PASS — no blocking issues found.
```

An additional ID-existence validation can be run with:

```bash
python3 utils/validate_submission.py --check-ids
```

---

# 19. Reproduction

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the pipeline from the project root.

### Training

Generate training candidates:

```bash
python3 src/blocking.py
```

Generate matching features:

```bash
python3 src/features.py
```

Train the classifier:

```bash
python3 src/train_model.py
```

### Test inference

Run:

```bash
python3 src/predict.py
```

The prediction script generates:

```text
dataset/test/matching_results.tsv
```

and the candidate database can be used to produce:

```text
output/candidate_pairs.tsv
```

The exact paths used by the scripts should be checked against the configuration constants in the submitted source files.

---

# 20. Computational Design

The dataset contains millions of records, so the implementation avoids Python-level nested loops over Source 1 × Source 2/3.

DuckDB is used for:

* TSV ingestion,
* normalized-key storage,
* blocking indexes,
* candidate generation,
* large joins,
* aggregation,
* Parquet export.

This keeps large relational operations inside a columnar query engine.

Feature generation uses compact numerical/string-derived features rather than expensive pairwise edit-distance calculations over the complete candidate set.

---

# 21. Limitations

The current system has several known limitations:

* The blocker can miss true matches when no selected blocking key is shared.
* The model does not use full edit-distance or TF-IDF similarity features in the final fast feature set.
* Address similarity is represented using exact/contains/first-token/number signals rather than a full semantic address model.
* The test set has no ground truth, so threshold selection is based on the held-out training validation split.
* Some Source 1 entities receive zero candidates from the blocker.

These limitations are accepted in exchange for keeping candidate generation and inference computationally practical at the dataset scale.

---

# 22. Fair-Play Compliance

The solution uses only the provided challenge data and locally generated features.

No external business database, business registry, geocoding service, commercial entity-resolution API, or external identity lookup is used.

The pipeline is based entirely on:

* provided TSV data,
* deterministic normalization,
* blocking keys,
* locally generated pair features,
* supervised training labels,
* the trained scikit-learn classifier.

---

# 23. Final Submission

The challenge final package contains:

```text
<team_name>_submission.zip
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
│
└── Documentation_template.md
```

The leaderboard prediction file is:

```text
output/matching_results.tsv
```

The candidate set is:

```text
output/candidate_pairs.tsv
```

---

# 24. Summary

The final architecture is:

```text
                 ┌─────────────────────┐
                 │   Raw TSV Sources   │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ Unicode Normalizer  │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ DuckDB Blocking     │
                 │ 6 blocking routes   │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ Candidate Pairs     │
                 │ 140.99M test pairs  │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ 16 Pair Features    │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ HistGradientBoosting│
                 │ Classifier          │
                 └──────────┬──────────┘
                            │
                            v
                 ┌─────────────────────┐
                 │ Threshold = 0.70    │
                 └──────────┬──────────┘
                            │
                            v
              ┌─────────────────────────────┐
              │ Final Entity Resolution     │
              └──────────────┬──────────────┘
                             │
                    ┌────────┴────────┐
                    v                 v
             matching_results    candidate_pairs
```

The system therefore separates the problem into a scalable retrieval stage and a precision-oriented supervised matching stage.
