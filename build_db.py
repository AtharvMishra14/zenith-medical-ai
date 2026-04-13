import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("1. Loading Medical Data...")
# Read the file directly from your VS Code folder
df = pd.read_csv('mtsamples.csv') 
df = df.dropna(subset=['transcription']) 

print("2. Chunking Text...")
documents = [Document(page_content=text, metadata={"specialty": specialty}) 
             for text, specialty in zip(df['transcription'], df['medical_specialty'])]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = text_splitter.split_documents(documents)
print(f"   -> Created {len(chunked_docs)} chunks!")

print("3. Initializing Embedding Model...")
# Your local GPU will help accelerate this
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("4. Building Vector Database (This might take a minute or two)...")
# Save it directly to a brand new folder to avoid confusion
vector_db = Chroma.from_documents(
    documents=chunked_docs, 
    embedding=embeddings, 
    persist_directory="./chroma_db_real" 
)

print("✅ SUCCESS: Real Vector database built locally!")