import openai
import uuid
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV
from pinecone import Pinecone, ServerlessSpec

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_temp_index() -> str:
    index_name = f"policy-{uuid.uuid4().hex[:8]}"
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    return index_name

def embed_chunks_store(index_name: str, chunks: list[str]):
    index = pc.Index(index_name)
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])
    print(f" store {len(chunks)}  index: {index_name}")

def get_embedding(text: str) -> list[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding
