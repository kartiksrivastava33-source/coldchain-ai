
import os
import re
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from langchain_core.tools import tool


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

# MySQL configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "coldchain_db")

MYSQL_USER = os.getenv("MYSQL_AGENT_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_AGENT_PASSWORD")

# Qdrant configuration
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "coldchain-sop"

# Embedding configuration
EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    os.getenv("Local_Embedding_Model", "BAAI/bge-m3")
)


# ==================================================
# 2. EMBEDDING MODEL
# ==================================================

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Loads BGE-M3 once and reuses the model.
    """

    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 8
        }
    )


# ==================================================
# 3. QDRANT VECTOR STORE
# ==================================================

@lru_cache(maxsize=1)
def get_vector_store():
    """
    Connects to the persistent Qdrant SOP collection.
    """

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY missing from .env")

    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
        check_compatibility=False
    )

    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run the SOP ingestion pipeline first."
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings()
    )


# ==================================================
# 4. MYSQL DATABASE CONNECTION
# ==================================================

@lru_cache(maxsize=1)
def get_mysql_engine():
    """
    Creates a reusable connection to MySQL
    using the dedicated read-only agent account.
    """

    if not MYSQL_USER or not MYSQL_PASSWORD:
        raise ValueError(
            "MYSQL_AGENT_USER or MYSQL_AGENT_PASSWORD "
            "is missing from .env"
        )

    database_url = URL.create(
        drivername="mysql+pymysql",
        username=MYSQL_USER,
        password=MYSQL_PASSWORD,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        database=MYSQL_DATABASE
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 10
        }
    )


# ==================================================
# 5. TOOL 1: SHIPMENT TELEMETRY
# ==================================================

@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a read-only MySQL SELECT query against
    the VW_ACTIVE_FLEET operational view.

    Available columns:
    Record_ID, Timestamp, Latitude, Longitude,
    Current_Temperature_C, Cargo_Condition_Code,
    Risk_Classification, Delay_Probability,
    Port_Congestion_Level, Route_Risk_Index.

    Use MySQL syntax, not SQL Server syntax.
    Maximum 20 records are returned.
    """

    query = sql_query.strip().rstrip(";").strip()

    if not query:
        return "Error: SQL query cannot be empty."

    # Only permit SELECT statements.
    if not re.match(r"^SELECT\b", query, re.IGNORECASE):
        return "SECURITY BLOCK: Only SELECT queries are permitted."

    # Reject multiple statements and SQL comments.
    if ";" in query or "--" in query or "/*" in query or "#" in query:
        return "SECURITY BLOCK: Multiple statements and SQL comments are prohibited."

    # Reject dangerous operations and execution functions.
    blocked_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP",
        "ALTER", "CREATE", "TRUNCATE", "REPLACE",
        "GRANT", "REVOKE", "CALL", "EXECUTE",
        "LOAD_FILE", "INTO OUTFILE", "INTO DUMPFILE"
    ]

    if re.search(
        r"\b(" + "|".join(blocked_keywords) + r")\b",
        query,
        re.IGNORECASE
    ):
        return "SECURITY BLOCK: Restricted SQL operation detected."

    # Add a result limit if the agent did not provide one.
    if not re.search(r"\bLIMIT\b", query, re.IGNORECASE):
        query += " LIMIT 20"

    # Cap simple LIMIT values at 20.
    limit_match = re.search(
        r"\bLIMIT\s+(\d+)\s*$",
        query,
        re.IGNORECASE
    )

    if limit_match:
        requested_limit = int(limit_match.group(1))
        safe_limit = min(requested_limit, 20)

        query = re.sub(
            r"\bLIMIT\s+\d+\s*$",
            f"LIMIT {safe_limit}",
            query,
            flags=re.IGNORECASE
        )

    try:
        engine = get_mysql_engine()

        with engine.connect() as connection:

            # Limit query execution time to 10 seconds.
            connection.execute(
                text("SET SESSION MAX_EXECUTION_TIME = 10000")
            )

            result = connection.execute(text(query))

            columns = list(result.keys())
            rows = result.fetchmany(20)

        if not rows:
            return "No shipment records matched the query."

        output = [
            f"COLUMNS: {', '.join(columns)}",
            f"RECORDS RETURNED: {len(rows)}"
        ]

        for row in rows:
            output.append(str(tuple(row)))

        return "\n".join(output)

    except Exception as error:
        return f"MySQL Telemetry Error: {str(error)}"


# ==================================================
# 6. TOOL 2: LIVE WEATHER
# ==================================================

@tool
def fetch_corridor_conditions(
    latitude: float,
    longitude: float
) -> str:
    """
    Fetches current weather information for shipment
    GPS coordinates using Open-Meteo.

    Returns temperature, humidity, wind speed and weather code.
    Weather observations are not traffic congestion measurements.
    """

    if not -90 <= latitude <= 90:
        return "Invalid latitude."

    if not -180 <= longitude <= 180:
        return "Invalid longitude."

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "wind_speed_10m,"
                    "weather_code"
                ),
                "timezone": "auto"
            },
            timeout=10
        )

        response.raise_for_status()

        payload = response.json()
        current = payload.get("current", {})

        if not current:
            return "No current weather information available."

        return (
            "--- LIVE CORRIDOR WEATHER ---\n"
            f"Latitude: {latitude}\n"
            f"Longitude: {longitude}\n"
            f"Observation Time: {current.get('time', 'not supplied')}\n"
            f"Temperature: {current.get('temperature_2m')} °C\n"
            f"Humidity: {current.get('relative_humidity_2m')}%\n"
            f"Wind Speed: {current.get('wind_speed_10m')} km/h\n"
            f"Weather Code: {current.get('weather_code')}\n"
            "Note: Weather does not establish traffic congestion.\n"
            "-----------------------------"
        )

    except requests.RequestException as error:
        return f"Weather API Error: {str(error)}"


# ==================================================
# 7. TOOL 3: SEMANTIC SOP RETRIEVAL
# ==================================================

@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches the Cold Chain SOP using BGE-M3
    embeddings and Qdrant vector similarity search.

    Retrieves relevant temperature limits, emergency
    cold-storage procedures, congestion rules and
    logistics risk escalation requirements.
    """

    if not query.strip():
        return "Please provide a compliance question."

    try:
        vector_store = get_vector_store()

        results = vector_store.similarity_search_with_score(
            query,
            k=3
        )

        if not results:
            return "No matching SOP clauses found."

        output = [
            "--- SEMANTIC SOP RETRIEVAL RESULTS ---",
            f"Query: {query}"
        ]

        for index, (document, score) in enumerate(
            results,
            start=1
        ):

            metadata = document.metadata

            output.append(
                f"\nRESULT {index}\n"
                f"Similarity Score: {score:.4f}\n"
                f"Source: {metadata.get('source_file', 'Unknown')}\n"
                f"Document Version: {metadata.get('version', 'Unknown')}\n"
                f"Content:\n{document.page_content}"
            )

        output.append("\n--------------------------------------")

        return "\n".join(output)

    except Exception as error:
        return f"Qdrant SOP Retrieval Error: {str(error)}"


# ==================================================
# 8. TOOL VERIFICATION
# ==================================================

if __name__ == "__main__":

    print("\n--- TEST 1: MYSQL TELEMETRY ---")

    print(
        query_telemetry_db.invoke(
            "SELECT COUNT(*) AS total_shipments "
            "FROM VW_ACTIVE_FLEET"
        )
    )

    print("\n--- TEST 2: QDRANT SOP RETRIEVAL ---")

    print(
        search_compliance_sop.invoke(
            "What should the dispatcher do when "
            "fresh perishables exceed 4 degrees Celsius?"
        )
    )

    print("\n--- TEST 3: LIVE WEATHER ---")

    print(
        fetch_corridor_conditions.invoke({
            "latitude": 40.375568,
            "longitude": -77.014317
        })
    )

    print("\nAll tool verification tests completed.")
