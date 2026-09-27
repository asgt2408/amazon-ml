import duckdb
import os
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(BASE, "dataset", "train")

S1_FILE = os.path.join(TRAIN, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN, "train_ground_truth.tsv")

OUTPUT = os.path.join(TRAIN, "candidate_pairs_validation_50k.tsv")
DB_FILE = os.path.join(TRAIN, "blocking_validation.duckdb")

S1_LIMIT = None
MAX_BLOCK = 100

START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", flush=True)


con = duckdb.connect(DB_FILE)

con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='8GB'")


def create_source(name, path, limit=None):

    limit_sql = f"LIMIT {limit}" if limit else ""

    log(f"Loading {name}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT
            entity_id,

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
            quote='',
            escape=''
        )
        {limit_sql}
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM {name}"
    ).fetchone()[0]

    log(f"{name}: {count:,} rows")


create_source("s1", S1_FILE, S1_LIMIT)
create_source("s2", S2_FILE)
create_source("s3", S3_FILE)


# ============================================================
# BASIC BLOCKING KEYS
#
# 1 = country + full name
# 2 = country + first + last token
# 3 = country + first + second token
# 4 = country + address number + first address token
# ============================================================

def create_basic_keys(source, output):

    log(f"Creating basic keys for {source}")

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
            regexp_extract(name, '([^ ]+)$', 1) AS block_key
        FROM {source}
        WHERE country <> ''
          AND name <> ''
          AND regexp_extract(name, '^([^ ]+)', 1) <> ''
          AND regexp_extract(name, '([^ ]+)$', 1) <> ''

        UNION ALL

        SELECT DISTINCT
            entity_id,
            3 AS key_type,
            country || '|' ||
            regexp_extract(name, '^([^ ]+)', 1) || '|' ||
            regexp_extract(name, '^([^ ]+) ([^ ]+)', 2) AS block_key
        FROM {source}
        WHERE country <> ''
          AND name <> ''
          AND regexp_extract(name, '^([^ ]+) ([^ ]+)', 2) <> ''

        UNION ALL

        SELECT DISTINCT
            entity_id,
            4 AS key_type,
            country || '|' ||
            regexp_extract(address, '[0-9]+', 0) || '|' ||
            regexp_extract(
                regexp_replace(address, '^[0-9]+[ ]*', ''),
                '^([^ ]+)',
                1
            ) AS block_key
        FROM {source}
        WHERE country <> ''
          AND address <> ''
          AND regexp_extract(address, '[0-9]+', 0) <> ''
          AND regexp_extract(
                regexp_replace(address, '^[0-9]+[ ]*', ''),
                '^([^ ]+)',
                1
              ) <> ''
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM {output}"
    ).fetchone()[0]

    log(f"{output}: {count:,} basic keys")


create_basic_keys("s1", "s1_keys")
create_basic_keys("s2", "s2_keys")
create_basic_keys("s3", "s3_keys")


# ============================================================
# COMBINED RARE NAME TOKENS
#
# Frequency is calculated across S2 + S3 together.
# This prevents common tokens from creating huge blocks.
#
# 5 = rare name token
# 6 = rare address token
# ============================================================

log("Building rare name tokens")

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

log("Building rare address tokens")

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

    log(f"Adding rare keys for {source}")

    con.execute(f"""
        INSERT INTO {output}

        SELECT DISTINCT
            s.entity_id,
            5 AS key_type,
            s.country || '|' || t.token AS block_key
        FROM {source} s
        CROSS JOIN UNNEST(string_split(s.name, ' ')) AS u(token)
        JOIN rare_name_tokens t
          ON t.country = s.country
         AND t.token = u.token
        WHERE s.country <> ''
          AND s.name <> ''

        UNION ALL

        SELECT DISTINCT
            s.entity_id,
            6 AS key_type,
            s.country || '|' || t.token AS block_key
        FROM {source} s
        CROSS JOIN UNNEST(string_split(s.address, ' ')) AS u(token)
        JOIN rare_address_tokens t
          ON t.country = s.country
         AND t.token = u.token
        WHERE s.country <> ''
          AND s.address <> ''
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM {output}"
    ).fetchone()[0]

    log(f"{output}: {count:,} total keys")


add_rare_keys("s1", "s1_keys")
add_rare_keys("s2", "s2_keys")
add_rare_keys("s3", "s3_keys")


def create_blocks(keys_table, source, output):

    log(f"Building blocks for {source}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {output} AS
        SELECT
            key_type,
            block_key,
            list(entity_id) AS entity_ids
        FROM {keys_table}
        GROUP BY key_type, block_key
        HAVING COUNT(*) <= {MAX_BLOCK}
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM {output}"
    ).fetchone()[0]

    log(f"{output}: {count:,} blocks")


create_blocks("s2_keys", "s2", "s2_blocks")
create_blocks("s3_keys", "s3", "s3_blocks")


log("Generating S2 candidates")

con.execute("""
    CREATE OR REPLACE TABLE candidate_s2 AS

    SELECT DISTINCT
        s1.entity_id AS source1_entity_id,
        unnest(b.entity_ids) AS candidate_entity_id
    FROM s1_keys s1
    JOIN s2_blocks b
      ON s1.key_type = b.key_type
     AND s1.block_key = b.block_key
""")

log("Generating S3 candidates")

con.execute("""
    CREATE OR REPLACE TABLE candidate_s3 AS

    SELECT DISTINCT
        s1.entity_id AS source1_entity_id,
        unnest(b.entity_ids) AS candidate_entity_id
    FROM s1_keys s1
    JOIN s3_blocks b
      ON s1.key_type = b.key_type
     AND s1.block_key = b.block_key
""")


log("Writing candidate file in batches")

if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

BATCH_SIZE = 100_000

total_s1 = con.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

first_batch = True

for start_id in range(0, total_s1, BATCH_SIZE):

    end_id = start_id + BATCH_SIZE

    log(f"Writing S1 rows {start_id:,} - {min(end_id, total_s1):,}")

    batch_file = OUTPUT + ".part"

    con.execute(f"""
        COPY (
            SELECT
                s.entity_id AS source1_entity_id,

                COALESCE(s2.candidate_s2_ids, '') AS candidate_s2_ids,

                COALESCE(s3.candidate_s3_ids, '') AS candidate_s3_ids

            FROM (
                SELECT
                    entity_id,
                    row_number() OVER (ORDER BY entity_id) AS rn
                FROM s1
            ) s

            LEFT JOIN (
                SELECT
                    source1_entity_id,
                    string_agg(
                        candidate_entity_id::VARCHAR,
                        ','
                    ) AS candidate_s2_ids
                FROM candidate_s2
                WHERE source1_entity_id IN (
                    SELECT entity_id
                    FROM (
                        SELECT
                            entity_id,
                            row_number() OVER (ORDER BY entity_id) AS rn
                        FROM s1
                    )
                    WHERE rn > {start_id}
                      AND rn <= {end_id}
                )
                GROUP BY source1_entity_id
            ) s2
              ON s.entity_id = s2.source1_entity_id

            LEFT JOIN (
                SELECT
                    source1_entity_id,
                    string_agg(
                        candidate_entity_id::VARCHAR,
                        ','
                    ) AS candidate_s3_ids
                FROM candidate_s3
                WHERE source1_entity_id IN (
                    SELECT entity_id
                    FROM (
                        SELECT
                            entity_id,
                            row_number() OVER (ORDER BY entity_id) AS rn
                        FROM s1
                    )
                    WHERE rn > {start_id}
                      AND rn <= {end_id}
                )
                GROUP BY source1_entity_id
            ) s3
              ON s.entity_id = s3.source1_entity_id

            WHERE s.rn > {start_id}
              AND s.rn <= {end_id}

        )
        TO '{batch_file}'
        (HEADER, DELIMITER '\\t')
    """)

    if first_batch:
        os.rename(batch_file, OUTPUT)
        first_batch = False
    else:
        with open(batch_file, "r") as f:
            lines = f.readlines()[1:]

        with open(OUTPUT, "a") as f:
            f.writelines(lines)

        os.remove(batch_file)

log("Candidate file completed")


log("Loading ground truth")

con.execute(f"""
    CREATE OR REPLACE TABLE gt AS
    SELECT
        source1_entity_id,
        matched_entity_ids
    FROM read_csv(
        '{GT_FILE}',
        delim='\\t',
        header=true,
        quote=''
    )
    WHERE source1_entity_id IN (
        SELECT entity_id FROM s1
    )
""")

con.execute("""
    CREATE OR REPLACE TABLE true_pairs AS

    SELECT
        source1_entity_id,
        trim(unnest(string_split(matched_entity_ids, ','))) AS entity_id
    FROM gt
    WHERE matched_entity_ids <> ''
""")

con.execute("""
    CREATE OR REPLACE TABLE all_candidates AS

    SELECT
        source1_entity_id,
        candidate_entity_id::VARCHAR AS entity_id
    FROM candidate_s2

    UNION

    SELECT
        source1_entity_id,
        candidate_entity_id::VARCHAR AS entity_id
    FROM candidate_s3
""")


true_count = con.execute("""
    SELECT COUNT(*)
    FROM true_pairs
""").fetchone()[0]

found_count = con.execute("""
    SELECT COUNT(*)
    FROM true_pairs t
    JOIN all_candidates c
      ON t.source1_entity_id = c.source1_entity_id
     AND t.entity_id = c.entity_id
""").fetchone()[0]

entities_with_true = con.execute("""
    SELECT COUNT(DISTINCT source1_entity_id)
    FROM true_pairs
""").fetchone()[0]

entities_with_any = con.execute("""
    SELECT COUNT(DISTINCT source1_entity_id)
    FROM true_pairs t
    WHERE EXISTS (
        SELECT 1
        FROM all_candidates c
        WHERE c.source1_entity_id = t.source1_entity_id
          AND c.entity_id = t.entity_id
    )
""").fetchone()[0]

entities_all = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            t.source1_entity_id,
            COUNT(*) AS true_count,
            SUM(
                CASE WHEN EXISTS (
                    SELECT 1
                    FROM all_candidates c
                    WHERE c.source1_entity_id = t.source1_entity_id
                      AND c.entity_id = t.entity_id
                )
                THEN 1 ELSE 0 END
            ) AS found_count
        FROM true_pairs t
        GROUP BY t.source1_entity_id
    )
    WHERE true_count = found_count
""").fetchone()[0]

avg_candidates = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            s.entity_id,
            COUNT(c.entity_id) AS cnt
        FROM s1 s
        LEFT JOIN all_candidates c
          ON s.entity_id = c.source1_entity_id
        GROUP BY s.entity_id
    )
""").fetchone()[0]

max_candidates = con.execute("""
    SELECT MAX(cnt)
    FROM (
        SELECT
            s.entity_id,
            COUNT(c.entity_id) AS cnt
        FROM s1 s
        LEFT JOIN all_candidates c
          ON s.entity_id = c.source1_entity_id
        GROUP BY s.entity_id
    )
""").fetchone()[0]

zero_candidates = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            s.entity_id,
            COUNT(c.entity_id) AS cnt
        FROM s1 s
        LEFT JOIN all_candidates c
          ON s.entity_id = c.source1_entity_id
        GROUP BY s.entity_id
        HAVING COUNT(c.entity_id) = 0
    )
""").fetchone()[0]


s1_count = con.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

s2_count = con.execute(
    "SELECT COUNT(*) FROM s2"
).fetchone()[0]

s3_count = con.execute(
    "SELECT COUNT(*) FROM s3"
).fetchone()[0]

candidate_count = con.execute("""
    SELECT COUNT(*)
    FROM all_candidates
""").fetchone()[0]

avg_candidates = con.execute("""
    SELECT AVG(cnt)
    FROM (
        SELECT
            s.entity_id,
            COUNT(c.entity_id) AS cnt
        FROM s1 s
        LEFT JOIN all_candidates c
          ON s.entity_id = c.source1_entity_id
        GROUP BY s.entity_id
    )
""").fetchone()[0]

zero_candidates = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            s.entity_id,
            COUNT(c.entity_id) AS cnt
        FROM s1 s
        LEFT JOIN all_candidates c
          ON s.entity_id = c.source1_entity_id
        GROUP BY s.entity_id
        HAVING COUNT(c.entity_id) = 0
    )
""").fetchone()[0]

print()
print("=" * 60)
print("FULL BLOCKING COMPLETE")
print("=" * 60)

print(f"S1 rows                  : {s1_count:,}")
print(f"S2 rows                  : {s2_count:,}")
print(f"S3 rows                  : {s3_count:,}")

print()
print(f"Total candidate pairs    : {candidate_count:,}")
print(f"Average candidates/S1    : {avg_candidates:.2f}")
print(f"Zero-candidate S1        : {zero_candidates:,}")

print()
print(f"Output                   : {OUTPUT}")
print(f"Runtime                  : {time.time() - START:.1f}s")

print("=" * 60)