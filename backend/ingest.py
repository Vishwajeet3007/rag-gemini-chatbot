import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "documents.pdf")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "faiss_index")

# -------------------------------------------------
# Ingest Function
# -------------------------------------------------
def ingest_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"PDF not found at: {DATA_PATH}")

    # 1. Load PDF
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)
    print(f"✅ Split into {len(docs)} chunks")

    # 3. Local embeddings (NO API, NO QUOTA)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 5. Save FAISS index
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index created and saved successfully")

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    ingest_documents()
