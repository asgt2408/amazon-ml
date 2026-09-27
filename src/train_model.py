import duckdb
import os
import time
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "dataset", "train")

FEATURE_FILE = os.path.join(TRAIN, "matching_features.parquet")
MODEL_FILE = os.path.join(TRAIN, "matching_model.joblib")

START = time.time()

con = duckdb.connect()

con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='7GB'")
con.execute("SET preserve_insertion_order=false")

print("Loading feature data...")

# ------------------------------------------------------------
# Split by SOURCE-1 entity
# This prevents the same S1 entity appearing in both
# training and validation.
# ------------------------------------------------------------

con.execute(f"""
    CREATE OR REPLACE TABLE data AS
    SELECT *
    FROM read_parquet('{FEATURE_FILE}')
""")

print("Creating entity split...")

con.execute("""
    CREATE OR REPLACE TABLE entity_split AS
    SELECT
        source1_entity_id,

        CASE
            WHEN hash(source1_entity_id) % 5 = 0
            THEN 'validation'
            ELSE 'train'
        END AS split

    FROM (
        SELECT DISTINCT source1_entity_id
        FROM data
    )
""")

FEATURES = [
    "country_match",
    "name_exact",
    "name_first_match",
    "name_last_match",
    "name_first_two_match",
    "name_contains",
    "name_length_diff",
    "name_length_ratio",
    "address_exact",
    "address_first_match",
    "address_number_match",
    "address_contains",
    "address_length_diff",
    "address_length_ratio",
    "name_missing",
    "address_missing",
]


feature_sql = ", ".join(FEATURES)

print("Loading training rows...")

train = con.execute(f"""
    SELECT
        {feature_sql},
        label

    FROM data d

    JOIN entity_split e
      ON d.source1_entity_id = e.source1_entity_id

    WHERE e.split = 'train'
""").fetchdf()

print(f"Training rows: {len(train):,}")

print("Loading validation rows...")

valid = con.execute(f"""
    SELECT
        {feature_sql},
        label

    FROM data d

    JOIN entity_split e
      ON d.source1_entity_id = e.source1_entity_id

    WHERE e.split = 'validation'
""").fetchdf()

print(f"Validation rows: {len(valid):,}")


X_train = train[FEATURES]
y_train = train["label"]

X_valid = valid[FEATURES]
y_valid = valid["label"]


print("Training HistGradientBoostingClassifier...")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)


print("Predicting validation set...")

probabilities = model.predict_proba(X_valid)[:, 1]


print()
print("=" * 70)
print("THRESHOLD SEARCH")
print("=" * 70)

best_threshold = 0.5
best_f05 = 0.0

for threshold in [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_valid,
        predictions,
        beta=0.5,
        zero_division=0
    )

    print(
        f"{threshold:5.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F0.5: {f05:.4f}"
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold


joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "threshold": best_threshold,
    },
    MODEL_FILE
)


print()
print("=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(f"Best threshold          : {best_threshold:.2f}")
print(f"Best validation F0.5    : {best_f05:.4f}")
print(f"Model                   : {MODEL_FILE}")
print(f"Runtime                 : {time.time() - START:.1f}s")

print("=" * 70)

con.close()