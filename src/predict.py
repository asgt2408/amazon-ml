import duckdb
import os
import time
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(BASE, "dataset", "test")
TRAIN = os.path.join(BASE, "dataset", "train")

S1_FILE = os.path.join(TEST, "test_source1.tsv")
S2_FILE = os.path.join(TEST, "test_source2.tsv")
S3_FILE = os.path.join(TEST, "test_source3.tsv")

MODEL_FILE = os.path.join(TRAIN, "matching_model.joblib")
OUTPUT = os.path.join(TEST, "matching_results.tsv")
FEATURE_FILE = os.path.join(TEST, "test_features.parquet")
DB_FILE = os.path.join(TEST, "test_matching.duckdb")

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


con = duckdb.connect(DB_FILE)

con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='7GB'")
con.execute("SET preserve_insertion_order=false")

def load_source(name, path):

    log(f"Loading {name}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT
            entity_id::VARCHAR AS entity_id,

            lower(trim(country)) AS country,

            trim(
                regexp_replace(
                    lower(business_name),
                    '[^\\p{{L}}\\p{{N}}]+',
                    ' ',
                    'g'
                )
            ) AS name,

            trim(
                regexp_replace(
                    lower(business_address),
                    '[^\\p{{L}}\\p{{N}}]+',
                    ' ',
                    'g'
                )
            ) AS address

        FROM read_csv(
            '{path}',
            delim='\\t',
            header=true,
            auto_detect=true,
            quote=''
        )
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM {name}"
    ).fetchone()[0]

    log(f"{name}: {count:,}")


load_source("s1", S1_FILE)
load_source("s2", S2_FILE)
load_source("s3", S3_FILE)


def create_basic_keys(source, output):

    con.execute(f"""
        CREATE OR REPLACE TABLE {output} AS

        SELECT DISTINCT
            entity_id,
            1 AS key_type,
            country || '|' || name AS block_key
        FROM {source}
        WHERE country <> ''
          AND name <> ''

        UNION ALL

        SELECT DISTINCT
            entity_id,
            2 AS key_type,
            country || '|' ||
            regexp_extract(name, '^([^ ]+)', 1) || '|' ||
            regexp_extract(name, '([^ ]+)$', 1)
        FROM {source}
        WHERE country <> ''
          AND name <> ''

        UNION ALL

        SELECT DISTINCT
            entity_id,
            3 AS key_type,
            country || '|' ||
            regexp_extract(name, '^([^ ]+)', 1) || '|' ||
            regexp_extract(
                name,
                '^([^ ]+) ([^ ]+)',
                2
            )
        FROM {source}
        WHERE country <> ''
          AND name <> ''
          AND regexp_extract(
                name,
                '^([^ ]+) ([^ ]+)',
                2
              ) <> ''

        UNION ALL

        SELECT DISTINCT
            entity_id,
            4 AS key_type,
            country || '|' ||
            regexp_extract(address, '[0-9]+', 0) || '|' ||
            regexp_extract(
                regexp_replace(
                    address,
                    '^[0-9]+[ ]*',
                    ''
                ),
                '^([^ ]+)',
                1
            )
        FROM {source}
        WHERE country <> ''
          AND address <> ''
          AND regexp_extract(
                address,
                '[0-9]+',
                0
              ) <> ''
    """)


create_basic_keys("s1", "s1_keys")
create_basic_keys("s2", "s2_keys")
create_basic_keys("s3", "s3_keys")


log("Building rare tokens")

con.execute("""
    CREATE OR REPLACE TABLE rare_name_tokens AS

    SELECT
        country,
        token

    FROM (
        SELECT
            country,
            unnest(string_split(name, ' ')) AS token
        FROM s2

        UNION ALL

        SELECT
            country,
            unnest(string_split(name, ' ')) AS token
        FROM s3
    )

    WHERE token <> ''
      AND length(token) >= 3
      AND token !~ '^[0-9]+$'

    GROUP BY country, token

    HAVING COUNT(*) BETWEEN 2 AND 100
""")


con.execute("""
    CREATE OR REPLACE TABLE rare_address_tokens AS

    SELECT
        country,
        token

    FROM (
        SELECT
            country,
            unnest(string_split(address, ' ')) AS token
        FROM s2

        UNION ALL

        SELECT
            country,
            unnest(string_split(address, ' ')) AS token
        FROM s3
    )

    WHERE token <> ''
      AND length(token) >= 3
      AND token !~ '^[0-9]+$'

    GROUP BY country, token

    HAVING COUNT(*) BETWEEN 2 AND 100
""")


def add_rare_keys(source, output):

    con.execute(f"""
        INSERT INTO {output}

        SELECT DISTINCT
            s.entity_id,
            5 AS key_type,
            s.country || '|' || u.token AS block_key

        FROM {source} s

        CROSS JOIN UNNEST(
            string_split(s.name, ' ')
        ) AS u(token)

        JOIN rare_name_tokens r
          ON r.country = s.country
         AND r.token = u.token

        UNION ALL

        SELECT DISTINCT
            s.entity_id,
            6 AS key_type,
            s.country || '|' || u.token AS block_key

        FROM {source} s

        CROSS JOIN UNNEST(
            string_split(s.address, ' ')
        ) AS u(token)

        JOIN rare_address_tokens r
          ON r.country = s.country
         AND r.token = u.token
    """)


add_rare_keys("s1", "s1_keys")
add_rare_keys("s2", "s2_keys")
add_rare_keys("s3", "s3_keys")

MAX_BLOCK = 100


def create_blocks(keys, output):

    con.execute(f"""
        CREATE OR REPLACE TABLE {output} AS

        SELECT
            key_type,
            block_key,
            list(entity_id) AS entity_ids

        FROM {keys}

        GROUP BY key_type, block_key

        HAVING COUNT(*) <= {MAX_BLOCK}
    """)


create_blocks("s2_keys", "s2_blocks")
create_blocks("s3_keys", "s3_blocks")


log("Generating test candidates")

con.execute("""
    CREATE OR REPLACE TABLE candidates AS

    SELECT DISTINCT
        k.entity_id AS source1_entity_id,
        unnest(b.entity_ids)::VARCHAR AS candidate_entity_id,
        's2' AS source

    FROM s1_keys k

    JOIN s2_blocks b
      ON k.key_type = b.key_type
     AND k.block_key = b.block_key

    UNION

    SELECT DISTINCT
        k.entity_id AS source1_entity_id,
        unnest(b.entity_ids)::VARCHAR AS candidate_entity_id,
        's3' AS source

    FROM s1_keys k

    JOIN s3_blocks b
      ON k.key_type = b.key_type
     AND k.block_key = b.block_key
""")


candidate_count = con.execute("""
    SELECT COUNT(*) FROM candidates
""").fetchone()[0]

log(f"Candidates: {candidate_count:,}")


log("Loading model")

saved = joblib.load(MODEL_FILE)

model = saved["model"]
features = saved["features"]
threshold = saved["threshold"]

log(f"Threshold: {threshold}")


con.execute("""
    CREATE OR REPLACE TABLE candidate_records AS

    SELECT
        entity_id,
        country,
        name,
        address,
        's2' AS source
    FROM s2

    UNION ALL

    SELECT
        entity_id,
        country,
        name,
        address,
        's3' AS source
    FROM s3
""")


log("Creating test features")

con.execute("""
    CREATE OR REPLACE TABLE test_features AS

    SELECT

        p.source1_entity_id,
        p.candidate_entity_id,
        p.source,

        CASE
            WHEN s1.country <> ''
             AND s1.country = c.country
            THEN 1 ELSE 0
        END AS country_match,

        CASE
            WHEN s1.name <> ''
             AND s1.name = c.name
            THEN 1 ELSE 0
        END AS name_exact,

        CASE
            WHEN regexp_extract(s1.name, '^([^ ]+)', 1)
               =
               regexp_extract(c.name, '^([^ ]+)', 1)
            THEN 1 ELSE 0
        END AS name_first_match,

        CASE
            WHEN regexp_extract(s1.name, '([^ ]+)$', 1)
               =
               regexp_extract(c.name, '([^ ]+)$', 1)
            THEN 1 ELSE 0
        END AS name_last_match,

        CASE
            WHEN regexp_extract(
                    s1.name,
                    '^([^ ]+ [^ ]+)',
                    1
                 )
                 =
                 regexp_extract(
                    c.name,
                    '^([^ ]+ [^ ]+)',
                    1
                 )
            THEN 1 ELSE 0
        END AS name_first_two_match,

        CASE
            WHEN s1.name <> ''
             AND c.name <> ''
             AND (
                strpos(s1.name, c.name) > 0
                OR strpos(c.name, s1.name) > 0
             )
            THEN 1 ELSE 0
        END AS name_contains,

        abs(
            length(s1.name) -
            length(c.name)
        ) AS name_length_diff,

        CASE
            WHEN greatest(
                length(s1.name),
                length(c.name)
            ) = 0
            THEN 0.0
            ELSE
                least(
                    length(s1.name),
                    length(c.name)
                )::DOUBLE
                /
                greatest(
                    length(s1.name),
                    length(c.name)
                )
        END AS name_length_ratio,

        CASE
            WHEN s1.address <> ''
             AND s1.address = c.address
            THEN 1 ELSE 0
        END AS address_exact,

        CASE
            WHEN regexp_extract(s1.address, '^([^ ]+)', 1)
               =
               regexp_extract(c.address, '^([^ ]+)', 1)
            THEN 1 ELSE 0
        END AS address_first_match,

        CASE
            WHEN regexp_extract(
                    s1.address,
                    '[0-9]+',
                    0
                 ) <> ''

             AND regexp_extract(
                    s1.address,
                    '[0-9]+',
                    0
                 )
                 =
                 regexp_extract(
                    c.address,
                    '[0-9]+',
                    0
                 )
            THEN 1 ELSE 0
        END AS address_number_match,

        CASE
            WHEN s1.address <> ''
             AND c.address <> ''
             AND (
                strpos(s1.address, c.address) > 0
                OR strpos(c.address, s1.address) > 0
             )
            THEN 1 ELSE 0
        END AS address_contains,

        abs(
            length(s1.address) -
            length(c.address)
        ) AS address_length_diff,

        CASE
            WHEN greatest(
                length(s1.address),
                length(c.address)
            ) = 0
            THEN 0.0
            ELSE
                least(
                    length(s1.address),
                    length(c.address)
                )::DOUBLE
                /
                greatest(
                    length(s1.address),
                    length(c.address)
                )
        END AS address_length_ratio,

        CASE
            WHEN s1.name = '' OR c.name = ''
            THEN 1 ELSE 0
        END AS name_missing,

        CASE
            WHEN s1.address = '' OR c.address = ''
            THEN 1 ELSE 0
        END AS address_missing

    FROM candidates p

    JOIN s1
      ON p.source1_entity_id = s1.entity_id

    JOIN candidate_records c
      ON p.candidate_entity_id = c.entity_id
     AND p.source = c.source
""")

log("Writing test features")

con.execute(f"""
    COPY test_features
    TO '{FEATURE_FILE}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

con.close()


log("Loading test features")

df = pd.read_parquet(FEATURE_FILE)

log(f"Feature rows: {len(df):,}")

X = df[features]

log("Predicting")

df["probability"] = model.predict_proba(X)[:, 1]

matches = df[
    df["probability"] >= threshold
].copy()


log("Creating final results")

matched = (
    matches
    .groupby("source1_entity_id")["candidate_entity_id"]
    .apply(
        lambda x: ",".join(
            sorted(set(x.astype(str)))
        )
    )
)

s1_ids = pd.read_csv(
    S1_FILE,
    sep="\t",
    header=0,
    usecols=["entity_id"],
    dtype=str
)

result = pd.DataFrame({
    "source1_entity_id": s1_ids["entity_id"].astype(str)
})

result["matched_entity_ids"] = (
    result["source1_entity_id"]
    .map(matched)
    .fillna("")
)

result.to_csv(
    OUTPUT,
    sep="\t",
    index=False
)


print()
print("=" * 60)
print("PREDICTION COMPLETE")
print("=" * 60)

print(f"Test S1 rows             : {len(result):,}")
print(f"Candidate pairs          : {candidate_count:,}")
print(f"Predicted candidate rows : {len(matches):,}")
print(f"S1 entities with match   : {len(matched):,}")
print(f"Threshold                : {threshold}")
print(f"Output                   : {OUTPUT}")
print(f"Runtime                  : {time.time() - START:.1f}s")

print("=" * 60)