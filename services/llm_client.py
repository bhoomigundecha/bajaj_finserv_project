from config import OPENAI_API_KEY, FINE_TUNED_MODEL, QDRANT_API_KEY, QDRANT_URL
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
import asyncio

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI()

TOP_K = 7  # increased from 5 to 10
SCORE_THRESHOLD = 0.4  # reduced from 0.6 to be more permissive
COLLECTION_NAME = "insurance-policies"

async def process_single_query(question: str, doc_id: str) -> str:
    try:
        print(f"\nRunning query: {question}")
        query_embed = await get_query_embedding(question)

        response = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embed,
            limit=TOP_K,
            score_threshold=SCORE_THRESHOLD,
            query_filter=Filter(
                must=[FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )]
            ),
            with_payload=True
        )

        print(f"Matches found: {len(response)}")
        for r in response:
            print(f" • Score: {r.score:.4f} → Chunk: {r.payload['text'][:80]}...")

        if not response:
            return "No relevant information found."

        context = "\n\n".join(match.payload["text"] for match in response)
        prompt = build_prompt(context, question)

        return await call_llm(prompt)

    except Exception as e:
        print(f" Exception in query: {e}")
        return f"Error: {str(e)}"

async def get_query_embedding(text: str) -> list[float]:
    def _get():
        return openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        ).data[0].embedding
    return await asyncio.to_thread(_get)

def build_prompt(context: str, question: str) -> str:
    return f"""You are a smart assistant trained to extract insurance-related details from policy documents.

Based on the [Context] provided, answer the [User Question] clearly and precisely.
If the answer is not directly present, reply with "No relevant information found."

[Context]:
{context}

[User Question]: {question}

[Answer]:"""

async def call_llm(prompt: str) -> str:
    def _call():
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to analyze insurance policies."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    return await asyncio.to_thread(_call)
