from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)
from config import QDRANT_API_KEY, QDRANT_URL
from more_itertools import chunked
from openai import OpenAI

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI()
COLLECTION_NAME = "insurance-policies"

def ensure_collection():
    print(" Ensuring Qdrant collection and index...")
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(" Creating collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    print(" Creating payload index on 'doc_id'...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD
    )

def embed_chunks_store(doc_id: str, chunks: list[str]):
    print(f" Embedding {len(chunks)} chunks for doc_id: {doc_id}")
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append(PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk, "doc_id": doc_id}
        ))

    print(f" Deleting existing vectors for doc_id: {doc_id}")
    
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )
    )

    print(" Upserting chunks to Qdrant...")
    for batch in chunked(vectors, 100):
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(" All chunks stored successfully.")

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding
