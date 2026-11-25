
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import URL


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dynamic_supply_chain_logistics_dataset.csv"
)

TABLE_NAME = "shipments"
VIEW_NAME = "VW_ACTIVE_FLEET"

DB_HOST = os.getenv("MYSQL_HOST", "localhost")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
DB_NAME = os.getenv("MYSQL_DATABASE", "coldchain_db")
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD")


# ==================================================
# 2. VALIDATE CONFIGURATION
# ==================================================

if not DB_PASSWORD:
    raise ValueError("MYSQL_PASSWORD is missing from .env")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")


# ==================================================
# 3. DATABASE CONNECTION
# ==================================================

database_url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_recycle=3600
)


# ==================================================
# 4. LOAD AND VALIDATE DATASET
# ==================================================

print("\nLoading shipment dataset...")

df = pd.read_csv(CSV_PATH)

required_columns = [
    "timestamp",
    "vehicle_gps_latitude",
    "vehicle_gps_longitude",
    "iot_temperature",
    "cargo_condition_status",
    "risk_classification",
    "delay_probability",
    "port_congestion_level",
    "route_risk_level"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("Dataset contains no records.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="raise"
)

# Create a stable identifier based on CSV row order.
df.insert(
    0,
    "record_id",
    range(1, len(df) + 1)
)

print("Dataset loaded:", len(df), "records")
print("Columns:", len(df.columns))


# ==================================================
# 5. CHECK EXISTING TABLE
# ==================================================

inspector = inspect(engine)

if inspector.has_table(TABLE_NAME):

    with engine.connect() as connection:
        existing_count = connection.execute(
            text(f"SELECT COUNT(*) FROM `{TABLE_NAME}`")
        ).scalar_one()

    if existing_count > 0:
        raise RuntimeError(
            f"Table '{TABLE_NAME}' already contains "
            f"{existing_count} records. "
            "Ingestion stopped to prevent accidental duplication "
            "or overwriting."
        )

    raise RuntimeError(
        f"Table '{TABLE_NAME}' already exists but is empty. "
        "Inspect it before proceeding."
    )


# ==================================================
# 6. IMPORT INTO MYSQL
# ==================================================

print("\nImporting records into MySQL...")

df.to_sql(
    name=TABLE_NAME,
    con=engine,
    if_exists="fail",
    index=False,
    chunksize=1000,
    method="multi"
)

print("Shipment records imported successfully.")


# ==================================================
# 7. CREATE OPERATIONAL SQL VIEW
# ==================================================

print("\nCreating operational shipment view...")

create_view_sql = """
CREATE OR REPLACE VIEW VW_ACTIVE_FLEET AS

SELECT
    record_id AS Record_ID,
    `timestamp` AS `Timestamp`,
    vehicle_gps_latitude AS Latitude,
    vehicle_gps_longitude AS Longitude,
    iot_temperature AS Current_Temperature_C,
    cargo_condition_status AS Cargo_Condition_Code,
    risk_classification AS Risk_Classification,
    delay_probability AS Delay_Probability,
    port_congestion_level AS Port_Congestion_Level,
    route_risk_level AS Route_Risk_Index

FROM shipments
"""

with engine.begin() as connection:
    connection.execute(text(create_view_sql))


# ==================================================
# 8. VERIFY INGESTION
# ==================================================

print("\nVerifying database records...")

with engine.connect() as connection:

    total_records = connection.execute(
        text("SELECT COUNT(*) FROM shipments")
    ).scalar_one()

    high_risk_records = connection.execute(
        text("""
            SELECT COUNT(*)
            FROM VW_ACTIVE_FLEET
            WHERE Risk_Classification = 'High Risk'
        """)
    ).scalar_one()

    sample_records = connection.execute(
        text("""
            SELECT *
            FROM VW_ACTIVE_FLEET
            LIMIT 5
        """)
    )

    print("\nTotal records:", total_records)
    print("High-risk records:", high_risk_records)

    print("\nSample records:")

    for record in sample_records:
        print(record)


# ==================================================
# 9. FINAL RESULT
# ==================================================

print("\n======================================")
print("MYSQL INGESTION COMPLETED")
print("======================================")
print("Database:", DB_NAME)
print("Table:", TABLE_NAME)
print("View:", VIEW_NAME)
print("Imported records:", total_records)
print("Expected records:", len(df))
print("======================================")

if total_records != len(df):
    raise RuntimeError("Record count verification failed.")

print("\nAll ingestion checks passed.")
