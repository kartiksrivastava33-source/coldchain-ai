
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

SOP_PATH = (
    PROJECT_ROOT
    / "data"
    / "policy"
    / "Cold_Chain_Incident_SOP_v2.md"
)

COLLECTION_NAME = "coldchain-sop"
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_SIZE = 1024

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY missing from .env")

if not SOP_PATH.exists():
    raise FileNotFoundError(f"SOP not found: {SOP_PATH}")


# ==========================================
# 2. CONNECT TO QDRANT
# ==========================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

print("Connected to Qdrant.")


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print(f"Loading embedding model: {EMBEDDING_MODEL}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 8
    }
)

print("Embedding model loaded.")


# ==========================================
# 4. LOAD SOP DOCUMENT
# ==========================================

sop_content = SOP_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=sop_content,
    metadata={
        "source_file": SOP_PATH.name,
        "document_type": "cold_chain_sop",
        "version": "2.4"
    }
)

print("SOP loaded successfully.")


# ==========================================
# 5. SPLIT INTO CHUNKS
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        " "
    ],
    add_start_index=True
)

chunks = text_splitter.split_documents([document])

print(f"Total chunks generated: {len(chunks)}")

if not chunks:
    raise ValueError("No SOP chunks generated.")


# ==========================================
# 6. CREATE QDRANT COLLECTION
# ==========================================

if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

    print("Qdrant collection created.")

else:
    print("Existing Qdrant collection found.")


# ==========================================
# 7. INITIALIZE VECTOR STORE
# ==========================================

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)


# ==========================================
# 8. GENERATE AND STORE EMBEDDINGS
# ==========================================

document_ids = []

for index, chunk in enumerate(chunks):

    identity = (
        f"{SOP_PATH.name}:"
        f"{chunk.metadata.get('start_index', index)}:"
        f"{chunk.page_content}"
    )

    document_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, identity)
    )

    document_ids.append(document_id)

print("Generating embeddings and indexing SOP...")

vector_store.add_documents(
    documents=chunks,
    ids=document_ids
)

print("SOP successfully indexed in Qdrant.")


# ==========================================
# 9. VERIFY COLLECTION
# ==========================================

collection_info = client.get_collection(
    COLLECTION_NAME
)

print("\n--- COLLECTION INFORMATION ---")
print("Collection:", COLLECTION_NAME)
print("Stored vectors:", collection_info.points_count)
print("Vector dimension:", VECTOR_SIZE)


# ==========================================
# 10. SEMANTIC RETRIEVAL TEST
# ==========================================

test_query = (
    "What should the dispatcher do "
    "when refrigerated cargo exceeds "
    "the permitted temperature?"
)

print("\n--- SEMANTIC RETRIEVAL TEST ---")
print("Query:", test_query)

results = vector_store.similarity_search_with_score(
    test_query,
    k=3
)

for index, (result, score) in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("Similarity score:", round(score, 4))
    print("Source:", result.metadata.get("source_file"))
    print("Content:")
    print(result.page_content)

print("\nSOP ingestion and retrieval test completed.")
