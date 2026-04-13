from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time

print("1. 📂 Loading the original DIRTY dataset (mtsamples.csv)...")
loader = CSVLoader(file_path="mtsamples.csv", encoding="utf-8")
dirty_docs = loader.load()
print(f"   - Successfully loaded {len(dirty_docs)} messy medical records.")

print("2. ✂️ Chunking the dirty data...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
dirty_chunks = text_splitter.split_documents(dirty_docs)
print(f"   - Sliced into {len(dirty_chunks)} vector chunks.")

print("3. 🧠 Connecting to the existing Massive Database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db_massive", embedding_function=embeddings)

print("4. 💉 Injecting the dirty data into the clean database (in batches)...")

# --- THE FIX: BATCH PROCESSING ---
# We cap the batch size at 5000 to safely stay under Chroma's 5461 limit
batch_size = 5000
total_chunks = len(dirty_chunks)

for i in range(0, total_chunks, batch_size):
    batch = dirty_chunks[i : i + batch_size]
    current_batch_num = (i // batch_size) + 1
    total_batches = (total_chunks // batch_size) + 1
    
    print(f"   - Processing batch {current_batch_num} of {total_batches} (Chunks {i} to {min(i + batch_size, total_chunks)})...")
    
    # Inject just this specific batch
    db.add_documents(batch)
    
    # Give the RTX 3050 a tiny 1-second breather between batches to prevent thermal throttling
    time.sleep(1) 

print("\n✅ SUCCESS! The Hybrid Database is complete and ready.")