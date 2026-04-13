from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("1. 📂 Loading the pristine, cleaned dataset...")
loader = CSVLoader(file_path="cleaned_massive_clinical_data.csv", encoding="utf-8")
docs = loader.load()
print(f"   - Successfully loaded {len(docs)} medical records.")

print("2. ✂️ Chunking the data for the AI's memory...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"   - Sliced the records into {len(chunks)} overlapping vector chunks.")

print("3. 🧠 Firing up the GPU for Mathematical Embeddings...")
# This uses the HuggingFace transformer model to convert text to high-dimensional math
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("4. 🏗️ Building the Massive Vector Database...")
print("   - PLEASE WAIT: Processing thousands of vectors. This will take a few minutes...")

# Notice we are saving this to a NEW folder: "chroma_db_massive"
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_massive"
)

print("\n✅ SUCCESS! The Enterprise-Grade Medical Database is locked and loaded.")