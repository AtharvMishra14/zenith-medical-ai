from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Connecting to local database...")
local_db = Chroma(
    persist_directory="./chroma_db_real", # Update this if you used Option B above!
    embedding_function=embeddings
)

# --- THE MAGIC TEST LINE ---
print(f"Total documents inside database: {local_db._collection.count()}")

print("\n--- Testing Medical Search ---")
query = "What are the common symptoms of a heart attack or cardiac arrest?"
results = local_db.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)