from dotenv import load_dotenv
import os

# ====================== IMPORTS FROM HELPER ======================
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)

load_dotenv()

# ====================== API KEYS ======================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ====================== LOAD & PROCESS DOCUMENTS ======================
print("📄 Loading PDFs...")
extracted_data = load_pdf_file(data='data/')

print("🔧 Filtering metadata...")
minimal_docs = filter_to_minimal_docs(extracted_data)

print("✂️ Splitting into chunks...")
text_chunks = text_split(minimal_docs)

print(f"✅ Created {len(text_chunks)} text chunks")

# ====================== EMBEDDINGS ======================
print("🔢 Loading embeddings model...")
embeddings = download_embeddings()

# ====================== PINECONE INDEX ======================
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medi-chatbot"

if not pc.has_index(index_name):
    print(f"🆕 Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,           # all-MiniLM-L6-v2 dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ====================== STORE IN PINECONE ======================
print("🚀 Storing vectors in Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("🎉 SUCCESS! Index created and stored in Pinecone.")
print(f"Index name: {index_name}")


# ====================== CHECK ACTUAL COUNT ======================
print("\n🔍 Checking Pinecone index stats...")
index = pc.Index(index_name)
stats = index.describe_index_stats()
print(stats)
print(f"✅ Total vectors stored: {stats.total_vector_count}")