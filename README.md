🩺 Zenith Medical AI: Clinical RAG Pipeline
Zenith Medical AI is a sophisticated Retrieval-Augmented Generation (RAG) system designed to perform semantic search and contextual analysis over clinical datasets (MTSamples). By leveraging vector embeddings and a local vector database, it allows users to query complex medical records using natural language.

🚀 Core Features
Semantic Retrieval: Uses ChromaDB to store and retrieve medical context based on vector similarity rather than simple keyword matching.

Massive Data Handling: Includes optimized scripts for building and indexing large-scale clinical datasets.

Streamlit Interface: A clean, interactive UI for real-time medical query processing.

Data Pipeline: Custom scripts for cleaning raw clinical CSV data (mtsamples.csv) and transforming it into searchable embeddings.

🛠️ Tech Stack
Language: Python 3.10+

Vector Database: ChromaDB

Frontend: Streamlit

Data Processing: Pandas, NumPy

Security: Environment-based configuration for API keys and sensitive credentials.

🏗️ MLOps & Model Factory Architecture
To ensure scalability and reproducibility, this project implements a Model Factory design pattern. This architecture decouples the data ingestion from the model generation, allowing for automated pipeline management.

Key Components:
Automated Pipeline: The factory logic automates the transformation of raw clinical records into vectorized embeddings without manual intervention.

Scalability: Designed to handle multiple specialized vector stores (e.g., separate indices for Radiology vs. General Surgery) by dynamically routing data through the Model Factory script.

Version Control for Data: While the massive binary files are ignored in Git, the factory ensures that any developer can reproduce the exact vector state using the provided cleaning and building scripts.

Modular Design: The separation of build_db.py and build_massive_db.py allows for "pluggable" data sources, making it easy to swap the local ChromaDB for a cloud-hosted solution like AWS Pinecone or RDS in the future.

📦 Installation & Setup
Clone the Repository:

Bash
git clone https://github.com/AtharvMishra14/zenith-medical-ai.git
cd zenith-medical-ai
Create Virtual Environment:

Bash
python -m venv venv
.\venv\Scripts\activate
Install Dependencies:

Bash
pip install -r requirements.txt
Build the Vector Database:
Note: Due to file size limits, the local vector store is not included in the repository. Generate it locally using:

Bash
python build_db.py
Run the Application:

Bash
streamlit run streamlit_app.py
📂 Project Structure
app.py: Main logic for the RAG pipeline.

build_db.py: Script to process CSV data and populate ChromaDB.

clean_data.py: Pre-processing scripts for the clinical dataset.

rag_test.py: Unit testing for retrieval accuracy.

🛡️ Security Note
This project follows best security practices by using .gitignore to prevent the accidental upload of AWS .pem keys, environment variables (.env), and heavy binary database files.