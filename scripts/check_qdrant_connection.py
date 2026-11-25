
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

api_key = os.getenv("QDRANT_API_KEY")

if not api_key:
    raise ValueError("QDRANT_API_KEY missing from .env")

client = QdrantClient(
    url="http://localhost:6333",
    api_key=api_key
)

# Check connection
print("Qdrant version:", client.info().version)

# Check collections
print("Collections:", client.get_collections())

collection_name = "coldchain_test"

# Create test collection
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": 1024,
            "distance": "Cosine"
        }
    )

print("Test collection created successfully.")

print(client.get_collection(collection_name))

# Clean up test collection
client.delete_collection(collection_name)

print("Qdrant connection test passed!")
