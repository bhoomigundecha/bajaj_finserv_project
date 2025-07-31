import openai
import asyncio
from config import (
    OPENAI_API_KEY,
    FINE_TUNED_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX,
)

from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI client (new API)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = None

# Create index if not exists
if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp" if your project uses that
            region=PINECONE_ENV
        )
    )

index = pc.Index(PINECONE_INDEX)

TOP_K = 5
NAMESPACE = "default"

def ensure_namespace_exists(index, namespace_name):
    """Ensure namespace exists by checking and creating if needed"""
    try:
        stats = index.describe_index_stats()
        if namespace_name not in stats.namespaces:
            # Create namespace with a dummy vector
            dummy_embedding = [0.0] * 1536
            index.upsert(
                vectors=[{"id": "init", "values": dummy_embedding, "metadata": {"text": "init"}}],
                namespace=namespace_name
            )
            print(f"Created namespace: {namespace_name}")
        else:
            print(f"Namespace {namespace_name} already exists")
    except Exception as e:
        print(f"Error checking namespace: {e}")

def process_multiquery(questions: list[str]) -> list[str]:
    # Ensure namespace exists before processing
    ensure_namespace_exists(index, NAMESPACE)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_all(questions))


async def process_all(questions: list[str]) -> list[str]:
    tasks = [process_single(q) for q in questions]
    return await asyncio.gather(*tasks)


async def process_single(question: str) -> str:
    try:
        query_embed = await get_query_embedding(question)
        response = index.query(
            vector=query_embed,
            top_k=TOP_K,
            include_metadata=True,
            namespace=NAMESPACE
        )
        
        if not response.get("matches"):
            return "No relevant information found in the database."
            
        context = "\n\n".join([match["metadata"]["text"] for match in response["matches"]])

        prompt = build_prompt(context, question)
        return await call_llm(prompt)

    except Exception as e:
        return f"Error: {str(e)}"


async def get_query_embedding(text: str) -> list[float]:
    def _get_embedding():
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    return await asyncio.to_thread(_get_embedding)


def build_prompt(context: str, question: str) -> str:
    return f"""You are a helpful assistant trained to extract information from insurance policies.

Use the provided document context to answer the user's question clearly, concisely, and accurately.

[Context]: 
{context}

[User Question]: 
{question}

[Answer]:"""


async def call_llm(prompt: str) -> str:
    def _call_llm():
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to analyze insurance policies."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    
    return await asyncio.to_thread(_call_llm)


# Data ingestion functions
def embed_chunks_store(chunks: list[str]):
    """Store document chunks with embeddings in Pinecone"""
    # Ensure namespace exists
    ensure_namespace_exists(index, NAMESPACE)
    
    # Optional: delete previous vectors in namespace
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
        print("Cleared existing vectors in namespace")
    except Exception as e:
        print(f"Note: Could not clear namespace (might be empty): {e}")

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    # Upsert in batches if large dataset
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")

    print(f"Successfully stored {len(chunks)} chunks in Pinecone")


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding