import os
from pathlib import Path
import time
from supabase import create_client, Client
from google import genai
from google.genai import types
import json
from dotenv import load_dotenv

# Finding #7 & #12: Load environment variables with path anchored to this file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Setup Supabase client - prefer service role key if available for administrative seeding
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://cqeubytgsrxdkfejxvan.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_KEY:
    print("Please set SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY environment variable.")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Setup Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Please set GEMINI_API_KEY environment variable.")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

# Load the comprehensive knowledge base
json_path = os.path.join(os.path.dirname(__file__), 'rag_knowledge_base.json')
with open(json_path, 'r', encoding='utf-8') as f:
    knowledge_data = json.load(f)

print(f"Loaded {len(knowledge_data)} documents from knowledge base. Generating embeddings...")

for i, entry in enumerate(knowledge_data):
    try:
        # Combine title, category, and content for rich context
        doc_text = f"[{entry['category']}] {entry['title']}: {entry['content']}"

        response = client.models.embed_content(
            model='models/gemini-embedding-2',
            contents=doc_text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        embedding = response.embeddings[0].values

        supabase.table('part_docs').insert({
            "content": doc_text,
            "embedding": embedding
        }).execute()

        print(f"Inserted document {i+1}")
        time.sleep(1)
    except Exception as e:
        print(f"Error processing document {i+1}: {e}")

print("Done! The RAG knowledge base is now populated.")
