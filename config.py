import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-1106:personal::Byb7huHp"

# Chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "insurance-index")
