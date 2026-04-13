import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup Page
st.set_page_config(page_title="Zenith Medical AI", page_icon="⚕️")
st.title("⚕️ Zenith Medical AI")
st.markdown("### Privacy-First Clinical RAG System")

# 2. Initialize Models (Using Groq for Speed on Free Tier)
# Get your FREE API key from https://console.groq.com/
GROQ_API_KEY = "YOUR_GROQ_API_KEY" 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db_massive", embedding_function=embeddings)
llm = ChatGroq(temperature=0.3, groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# 3. RAG Setup (Same strict prompt we perfected)
system_prompt = (
    "You are an expert Clinical AI Assistant. Use the provided Context to answer.\n\n"
    "### 🩺 Symptom Analysis\n[User symptoms only]\n\n"
    "### 📋 Potential Conditions\n[Diagnoses from context]\n\n"
    "### 💡 Recommended Next Steps\n[Clinical advice]\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# 4. Build Chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), combine_docs_chain)

# 5. UI Logic
user_input = st.text_input("Enter Patient Symptoms:")
if user_input:
    with st.spinner("Analyzing Clinical Records..."):
        response = rag_chain.invoke({"input": user_input})
        st.markdown(response["answer"])
        
        with st.expander("📄 View Clinical Sources"):
            for doc in response["context"]:
                st.write(doc.page_content[:300] + "...")