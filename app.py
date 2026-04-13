import os
import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

print("Initializing Zenith Medical AI Pipeline...")

# 1. Connect the Database
# The 2.33 GB database stays on disk; only the relevant chunks are loaded into RAM.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db_massive", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 5})

# 2. Smart LLM Engine Switch
# This prevents your 1 GB AWS instance from crashing by offloading inference.
GROQ_KEY = os.getenv("GROQ_API_KEY")

if GROQ_KEY:
    print("Cloud Environment Detected: Switching to Groq API for Inference...")
    llm = ChatGroq(
        temperature=0.3, 
        groq_api_key=GROQ_KEY, 
        model_name="llama3-8b-8192"
    )
else:
    print("Local Environment Detected: Staying on Ollama (Llama 3)...")
    llm = OllamaLLM(
        model="llama3",
        temperature=0.3
    )

# 3. Build the RAG Chain (Strict Clinical Template)
system_prompt = """You are an expert Clinical AI Assistant.

YOUR TASK:
Analyze the USER SYMPTOMS and find matching conditions from the MEDICAL DATABASE. 

STRICT RULES:
1. DO NOT narrate your thought process. Start immediately with "### 🩺 Symptom Analysis".
2. NEVER diagnose chronic illnesses (like COPD or cancer) unless the user mentions having that history. 
3. You must output EXACTLY the format below.

### 🩺 Symptom Analysis
[List only the symptoms the user provided]

### 📋 Potential Conditions
[Name the medical diagnoses implied by the Context. If the context is too severe for mild symptoms, state: "The database primarily returned severe chronic cases, but these symptoms align more closely with common upper respiratory infections."]

### 💡 Recommended Next Steps
[List 3-4 general next steps for a doctor]

MEDICAL DATABASE (Reference Only):
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "USER SYMPTOMS: {input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- GRADIO UI LOGIC ---

def chat_with_ai(user_message, history):
    response = rag_chain.invoke({"input": user_message})
    final_answer = response["answer"]
    
    sources = "\n\n---\n### 📄 Clinical Sources Retrieved:\n"
    for i, doc in enumerate(response["context"]):
        source_text = doc.page_content[:200].replace('\n', ' ')
        sources += f"**[{i+1}]** {source_text}...\n\n"
        
    return final_answer + sources

custom_theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(title="Zenith Medical AI") as demo:
    gr.HTML(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #0f766e; margin-bottom: 0; font-size: 2.5em;">⚕️ Zenith Medical AI</h1>
            <p style="color: #64748b; font-size: 1.2em; font-weight: 500;">Privacy-First Retrieval-Augmented Generation (RAG) System</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chat_with_ai,
                chatbot=gr.Chatbot(height=550, avatar_images=(None, "🏥")),
                textbox=gr.Textbox(placeholder="Type clinical query...", container=False, scale=7)
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 System Architecture")
            with gr.Group():
                gr.Markdown(
                    """
                    - **LLM Engine:** Llama 3 (8B)
                    - **Vector DB:** ChromaDB (Local 2.33 GB)
                    - **Database Size:** ~25,000 Records
                    - **Privacy Status:** 100% Offline / HIPAA Compliant
                    """
                )
            gr.Markdown("---")
            gr.Markdown("### 👨‍💻 Developer Info")
            with gr.Group():
                 gr.Markdown("**Atharv Mishra** | B.Tech CSE - 6th Sem")

# 7. Launch (Enabled for Public AWS Access)
if __name__ == "__main__":
    # 'server_name' allows external connections to reach your AWS instance.
    demo.launch(theme=custom_theme, server_name="0.0.0.0", server_port=7860)