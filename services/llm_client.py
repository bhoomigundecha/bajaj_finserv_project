from config import OPENAI_API_KEY, FINE_TUNED_MODEL, PINECONE_API_KEY
from pinecone import Pinecone
import openai
import asyncio

openai.api_key = OPENAI_API_KEY
client = openai
pc = Pinecone(api_key=PINECONE_API_KEY)
TOP_K = 5

async def process_single_query(question: str, index_name: str, namespace: str = "") -> str:
    RELEVANCE_THRESHOLD = 0.75
    DEBUG_MATCH_SCORES = False

    try:
        query_embed = await get_query_embedding(question)

        index = pc.Index(index_name)  # ✅ fix: define index here
        response = index.query(
            vector=query_embed,
            top_k=TOP_K,
            include_metadata=True,
            namespace=namespace
        )

        matches = response.get("matches", [])
        if DEBUG_MATCH_SCORES:
            for m in matches:
                print(f"Score: {m['score']:.4f} | Snippet: {m['metadata']['text'][:80]}")

        relevant_matches = [m for m in matches if m['score'] >= RELEVANCE_THRESHOLD]
        if not relevant_matches and matches:
            relevant_matches = matches[:1]

        if not relevant_matches:
            return "No relevant information found."

        context = "\n\n".join(m["metadata"]["text"] for m in relevant_matches)
        prompt = build_prompt(context, question)

        return await call_llm(prompt)

    except Exception as e:
        return f"Error: {str(e)}"

async def get_query_embedding(text: str) -> list[float]:
    def _get():
        return client.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
    return await asyncio.to_thread(_get)

def build_prompt(context: str, question: str) -> str:
    return f"""You are a smart assistant trained to extract insurance-related details from policy documents.

Based on the [Context] provided, answer the [User Question] clearly and precisely. If the answer is not directly present, reply with "No relevant information found."

[Context]:
{context}

[User Question]: {question}

[Answer]:"""

async def call_llm(prompt: str) -> str:
    def _call():
        response = client.ChatCompletion.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to analyze insurance policies."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response["choices"][0]["message"]["content"].strip()

    return await asyncio.to_thread(_call)
