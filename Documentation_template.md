# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TEAM-AstRa

**Team Members:** ASHU GUPTA, ATHARV RAWAL, NISHANT KUMAR JHA, SUBRATA SAHU

**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We developed a scalable two-stage business entity resolution pipeline consisting of **multi-key blocking followed by supervised candidate matching**. The system uses Unicode-safe normalization, DuckDB-based candidate generation, lightweight name/address similarity features, and a `HistGradientBoostingClassifier` to identify matching Source 2/Source 3 records for each Source 1 entity.

The final test pipeline generated **140,995,004 candidate pairs** for 1,732,544 Source 1 entities and produced 4,590,539 predicted matching candidate rows using a probability threshold of 0.70. The best measured validation F0.5 was **0.8962**.

---

## 2. Methodology

### 2.1 Problem Analysis

The challenge contains three independent business-record sources with no common cross-source identifier.

Source 1 is the reference entity set, and each Source 1 entity can have zero, one, or multiple matching records in Source 2 and Source 3.

During preprocessing and data inspection, the following noise patterns were considered:

* Business-name case differences.
* Punctuation and whitespace differences.
* Legal suffix variations such as `Corp`, `Corporation`, `Ltd`, `Limited`, `Inc`, etc.
* Business-name abbreviation differences.
* Different ordering or partial representation of name tokens.
* Address abbreviations such as street/road variations.
* Partial addresses.
* Missing address components.
* Different address formatting.
* Numeric address components.
* Missing business names or addresses.
* Multilingual and Unicode text.

The normalization pipeline therefore preserves Unicode characters instead of converting all text to ASCII.

The challenge also contains an open-set country field. The pipeline treats country as a normalized string rather than restricting it to a fixed training-country list.

---

### 2.2 Solution Strategy

**Approach Type:** Blocking + Supervised Classifier

**Core Innovation:** A multi-route blocking system combines exact and structured business-name/address keys with rare-token blocking. DuckDB is used to perform large-scale candidate generation efficiently, while a lightweight gradient-boosting model performs the final candidate-pair classification.

The complete pipeline is:

```text
Raw Sources
    ↓
Unicode-safe Normalization
    ↓
Multi-key Blocking
    ↓
Candidate Pairs
    ↓
Pair Feature Generation
    ↓
HistGradientBoostingClassifier
    ↓
Probability Threshold = 0.70
    ↓
Final Entity Matches
```

The design separates **candidate recall** from **final matching precision**. Blocking determines which records are eligible for matching, while the classifier determines which candidates are finally accepted.

---

## 3. Candidate Generation (Blocking)

Blocking was required because directly comparing every Source 1 record against every Source 2 and Source 3 record would result in an infeasible all-pairs search.

The blocking implementation is in:

```text
src/blocking.py
```

### Blocking keys used

The final blocker uses multiple complementary keys.

#### 1. Country + exact normalized business name

```text
country + normalized_full_name
```

This captures strong exact-name matches while limiting collisions by country.

#### 2. Country + first and last name tokens

```text
country + first_name_token + last_name_token
```

This provides resilience to variations in the middle of business names.

#### 3. Country + first two name tokens

```text
country + first_name_token + second_name_token
```

This provides another name-based retrieval path.

#### 4. Country + address number + first address token

```text
country + address_number + first_address_token
```

This combines a strong address-number signal with a location token.

#### 5. Rare name tokens

Rare business-name tokens are indexed separately.

Only tokens within a bounded frequency range are used so that extremely common words do not create excessively large candidate blocks.

#### 6. Rare address tokens

Rare address tokens provide an additional candidate-generation route when business-name representations differ.

### Block-size control

Large blocks are restricted to approximately 100 records per indexed blocking group. This prevents common tokens from producing extremely large candidate sets.

---

### Candidate pairs generated

For the final test run:

| Dataset  |   Records |
| -------- | --------: |
| Source 1 | 1,732,544 |
| Source 2 | 4,887,273 |
| Source 3 | 5,082,316 |

The blocker generated:

**140,995,004 candidate pairs**

This corresponds to approximately **81.36 candidates per Source 1 entity on average**.

There were 3,763 Source 1 entities for which blocking generated no candidates. These entities are still included in the final `candidate_pairs.tsv` with an empty candidate list.

---

### How you ensured true matches were not lost

No single blocking key is relied upon.

Instead, multiple independent blocking routes are combined:

* exact normalized name,
* structured name-token combinations,
* address-number/address-token combinations,
* rare name tokens,
* rare address tokens.

This gives records multiple opportunities to enter the candidate set when one representation differs because of noise.

The blocker was evaluated on the training data against the available ground truth before being applied to the test set.

On the 50k training validation subset used during blocker development, the final blocking approach achieved approximately:

* **83.26% pair recall**
* **96.40% entity-level any-match recall**

The final candidate set was then generated using the same blocking logic on the full test data.

---

## 4. Matching Model

### Features used

The final model uses 16 lightweight pair-level features.

#### Name features

* `name_exact`
* `name_first_match`
* `name_last_match`
* `name_first_two_match`
* `name_contains`
* `name_length_diff`
* `name_length_ratio`
* `name_missing`

These capture both exact and partial business-name agreement.

#### Address features

* `address_exact`
* `address_first_match`
* `address_number_match`
* `address_contains`
* `address_length_diff`
* `address_length_ratio`
* `address_missing`

These capture exact address agreement, address-number agreement, partial overlap and structural similarity.

#### Other

* `country_match`

Country agreement provides an additional discriminative signal.

The final feature set deliberately avoids expensive pairwise edit-distance calculations across the entire 140M+ candidate test set so that inference remains computationally practical.

---

### Model type

The final model is:

```text
sklearn.ensemble.HistGradientBoostingClassifier
```

The model was trained using the generated training candidate pairs.

The training feature dataset contained:

```text
24,093,709 candidate pairs
6,351,170 positive pairs
17,742,539 negative pairs
```

The validation split was performed at the Source 1 entity level rather than randomly splitting individual candidate pairs.

The validation rule used was:

```text
hash(source1_entity_id) % 5 == 0
```

This reduces leakage caused by placing candidate pairs belonging to the same Source 1 entity into both training and validation sets.

---

### Threshold selection method

The threshold was selected by evaluating multiple probability thresholds on the validation set using F0.5.

The best measured validation result was:

| Threshold |  Precision |     Recall |       F0.5 |
| --------: | ---------: | ---------: | ---------: |
|      0.50 |     0.9229 |     0.7536 |     0.8832 |
|      0.55 |     0.9376 |     0.7370 |     0.8892 |
|      0.60 |     0.9458 |     0.7271 |     0.8921 |
|      0.65 |     0.9535 |     0.7153 |     0.8939 |
|  **0.70** | **0.9715** | **0.6842** | **0.8962** |
|      0.75 |     0.9787 |     0.6682 |     0.8955 |
|      0.80 |     0.9840 |     0.6536 |     0.8936 |

Therefore, the final inference threshold was set to:

```text
0.70
```

---

## 5. Results & Error Analysis

### F_0.5 Score (macro)

**Best validation F0.5: 0.8962**

This was obtained using a probability threshold of 0.70 on the entity-level validation split.

The test set does not provide ground-truth labels, so a test F0.5 cannot be calculated locally.

---

### Final test statistics

The completed test inference produced:

```text
Test S1 rows             : 1,732,544
Candidate pairs          : 140,995,004
Predicted candidate rows : 4,590,539
S1 entities with match   : 1,511,135
S1 entities without match: 221,409
Threshold                : 0.70
```

---

### Common false positives (wrong merges)

Based on the structure of the features and the observed validation behavior, likely false-positive patterns include:

* Different businesses sharing a common or generic business name.
* Businesses with similar names within the same country.
* Common address tokens producing similar address representations.
* Records sharing an address number but representing different businesses.
* Short business names where small textual similarities are less discriminative.

The precision-oriented threshold of 0.70 was selected to reduce these false merges relative to lower thresholds.

---

### Common false negatives (missed matches)

Likely false-negative patterns include:

* Strongly corrupted business names.
* Business names with substantial word-order changes.
* Records with very incomplete addresses.
* Missing names or addresses.
* Transliteration differences that are not captured by the current deterministic normalization.
* True matches that do not share any of the blocking keys.
* Records whose useful similarity requires more expensive character-level or semantic similarity features.

A major limitation is that a true match not included in the blocking candidate set cannot be recovered by the matching model.

---

## 6. Conclusion

The solution combines Unicode-safe normalization, multi-route blocking, DuckDB-based large-scale candidate generation, lightweight pair-level features, and a supervised gradient-boosting classifier. The final system generated 140.99M candidates for the full test set while maintaining a compact candidate set per Source 1 entity and achieved a best validation F0.5 of 0.8962.

The main lesson from the implementation was that scalable entity resolution requires separating high-recall candidate generation from precision-oriented final matching rather than attempting an unrestricted comparison across millions of records.

---

## Appendix

### A. Code Artefacts

The complete runnable pipeline is included in:

```text
code/business_entity_resolution/
```

The source code is organized under:

```text
code/business_entity_resolution/
└── src/
    ├── normalize.py
    ├── blocking.py
    ├── features.py
    ├── train_model.py
    ├── predict.py
    └── test_blocking.py
```

Supporting files include:

```text
code/business_entity_resolution/
├── README.md
└── requirements.txt
```

The main pipeline stages are:

```text
normalize.py
    ↓
blocking.py
    ↓
features.py
    ↓
train_model.py
    ↓
predict.py
```

The final prediction stage produces:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

The trained classifier is stored using `joblib`.

The implementation uses:

* Python
* DuckDB
* pandas
* NumPy
* scikit-learn
* PyArrow
* joblib

---

### B. Additional Results

#### Training candidate-generation validation

The final blocker developed on the training validation subset produced:

```text
Candidate pairs           : 144,229 true pairs recovered
Pair recall               : 83.26%
Entity any-match recall   : 96.40%
```

The blocker generated approximately 73 candidates per Source 1 entity on that validation subset.

#### Full test candidate generation

```text
Source 1 records          : 1,732,544
Source 2 records          : 4,887,273
Source 3 records          : 5,082,316
Candidate pairs            : 140,995,004
Average candidates/S1     : ~81.36
Zero-candidate S1         : 3,763
```

#### Final matching

```text
Predicted candidate rows   : 4,590,539
S1 entities with matches   : 1,511,135
S1 entities without match  : 221,409
Decision threshold         : 0.70
```

#### Submission validation

The final submission files were validated using the provided challenge validator.

```text
matching_results.tsv       : 1,732,544 rows
candidate_pairs.tsv        : 1,732,544 rows

PASS — no blocking issues found.
```

The final package contains both required output files together with the runnable pipeline and this methodology document.
