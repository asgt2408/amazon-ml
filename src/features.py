import duckdb
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "dataset", "train")

DB_FILE = os.path.join(TRAIN, "blocking_validation.duckdb")
OUTPUT = os.path.join(TRAIN, "matching_features.parquet")

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


con = duckdb.connect(DB_FILE)

con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='7GB'")
con.execute("SET preserve_insertion_order=false")


positive_count = con.execute("""
    SELECT COUNT(*) FROM positives
""").fetchone()[0]

negative_count = con.execute("""
    SELECT COUNT(*) FROM negatives
""").fetchone()[0]

training_count = positive_count + negative_count

log(f"Positive pairs: {positive_count:,}")
log(f"Negative pairs: {negative_count:,}")
log(f"Training pairs: {training_count:,}")


log("Preparing candidate records")

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


if os.path.exists(OUTPUT):
    os.remove(OUTPUT)


log("Calculating fast features")

query = """
SELECT

    p.source1_entity_id,
    p.candidate_entity_id,
    p.source,
    p.label,

    ------------------------------------------------------------
    -- COUNTRY
    ------------------------------------------------------------

    CASE
        WHEN s1.country <> ''
         AND s1.country = c.country
        THEN 1
        ELSE 0
    END AS country_match,


    ------------------------------------------------------------
    -- NAME
    ------------------------------------------------------------

    CASE
        WHEN s1.name <> ''
         AND s1.name = c.name
        THEN 1
        ELSE 0
    END AS name_exact,


    -- First name token
    CASE
        WHEN regexp_extract(s1.name, '^([^ ]+)', 1) <> ''
         AND regexp_extract(s1.name, '^([^ ]+)', 1)
             =
             regexp_extract(c.name, '^([^ ]+)', 1)
        THEN 1
        ELSE 0
    END AS name_first_match,


    -- Last name token
    CASE
        WHEN regexp_extract(s1.name, '([^ ]+)$', 1) <> ''
         AND regexp_extract(s1.name, '([^ ]+)$', 1)
             =
             regexp_extract(c.name, '([^ ]+)$', 1)
        THEN 1
        ELSE 0
    END AS name_last_match,


    -- First two tokens
    CASE
        WHEN regexp_extract(
                s1.name,
                '^([^ ]+ [^ ]+)',
                1
             ) <> ''
         AND regexp_extract(
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
        THEN 1
        ELSE 0
    END AS name_first_two_match,


    -- Name contains the other name
    CASE
        WHEN s1.name <> ''
         AND c.name <> ''
         AND (
             strpos(s1.name, c.name) > 0
             OR strpos(c.name, s1.name) > 0
         )
        THEN 1
        ELSE 0
    END AS name_contains,


    ------------------------------------------------------------
    -- NAME LENGTH
    ------------------------------------------------------------

    abs(
        length(s1.name) - length(c.name)
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


    ------------------------------------------------------------
    -- ADDRESS
    ------------------------------------------------------------

    CASE
        WHEN s1.address <> ''
         AND s1.address = c.address
        THEN 1
        ELSE 0
    END AS address_exact,


    -- Address first token
    CASE
        WHEN regexp_extract(s1.address, '^([^ ]+)', 1) <> ''
         AND regexp_extract(s1.address, '^([^ ]+)', 1)
             =
             regexp_extract(c.address, '^([^ ]+)', 1)
        THEN 1
        ELSE 0
    END AS address_first_match,


    -- Address number
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

        THEN 1
        ELSE 0
    END AS address_number_match,


    -- Address contains the other address
    CASE
        WHEN s1.address <> ''
         AND c.address <> ''
         AND (
             strpos(s1.address, c.address) > 0
             OR strpos(c.address, s1.address) > 0
         )
        THEN 1
        ELSE 0
    END AS address_contains,


    ------------------------------------------------------------
    -- ADDRESS LENGTH
    ------------------------------------------------------------

    abs(
        length(s1.address) - length(c.address)
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


    ------------------------------------------------------------
    -- MISSING DATA
    ------------------------------------------------------------

    CASE
        WHEN s1.name = '' OR c.name = ''
        THEN 1
        ELSE 0
    END AS name_missing,


    CASE
        WHEN s1.address = '' OR c.address = ''
        THEN 1
        ELSE 0
    END AS address_missing


FROM (

    SELECT
        source1_entity_id,
        candidate_entity_id,
        source,
        label
    FROM positives

    UNION ALL

    SELECT
        source1_entity_id,
        candidate_entity_id,
        source,
        label
    FROM negatives

) p

JOIN s1
  ON p.source1_entity_id = s1.entity_id

JOIN candidate_records c
  ON p.candidate_entity_id = c.entity_id
 AND p.source = c.source
"""


log("Writing features directly to Parquet")

con.execute(f"""
    COPY (
        {query}
    )
    TO '{OUTPUT}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
""")

rows = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{OUTPUT}')
""").fetchone()[0]

columns = con.execute(f"""
    DESCRIBE
    SELECT *
    FROM read_parquet('{OUTPUT}')
""").fetchall()


print()
print("=" * 60)
print("FEATURE GENERATION COMPLETE")
print("=" * 60)

print(f"Positive pairs           : {positive_count:,}")
print(f"Negative pairs           : {negative_count:,}")
print(f"Training pairs           : {training_count:,}")
print(f"Feature rows             : {rows:,}")
print(f"Feature columns          : {len(columns)}")
print(f"Output                   : {OUTPUT}")
print(f"Runtime                  : {time.time() - START:.1f}s")

print()
print("Features:")
for col in columns:
    print(f"  - {col[0]}")

print("=" * 60)

con.close()