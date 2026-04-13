from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# --- THE TWO UPDATED LINES BELOW ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
# -----------------------------------

print("1. Waking up the Librarian (Vector DB)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db_real", embedding_function=embeddings)

# Tell the database to retrieve the top 3 most relevant medical snippets
retriever = db.as_retriever(search_kwargs={"k": 3})

print("2. Paging the Doctor (Llama 3)...")
llm = OllamaLLM(model="llama3")

print("3. Setting up the clinical instructions...")
# This prompt forces the AI to only use your medical data, preventing hallucinations
system_prompt = (
    "You are a highly precise medical AI assistant. "
    "Use the following pieces of retrieved medical context to answer the user's question. "
    "If you don't know the answer based on the context, say that you don't know. "
    "Keep the answer concise, professional, and easy to understand.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Assemble the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n--- 🩺 Testing the Medical AI ---")
query = "What are the common symptoms of a heart attack or cardiac arrest?"
print(f"User Question: {query}\n")
print("AI is reading the files and typing a response...\n")

# Run the query through the pipeline!
response = rag_chain.invoke({"input": query})

print("🤖 AI ANSWER:")
print(response["answer"])