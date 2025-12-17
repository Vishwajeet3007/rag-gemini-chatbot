import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# -------------------------------------------------
# Load env
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "faiss_index")

# -------------------------------------------------
# Embeddings (same as ingest.py)
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# Load FAISS
# -------------------------------------------------
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------
# Local LLM (NO API, NO QUOTA)
# -------------------------------------------------
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -------------------------------------------------
# Prompt
# -------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# ✅ EXPORTED FUNCTION
# -------------------------------------------------
def rag_app(question: str) -> str:
    docs = retriever.invoke(question)
    context = format_docs(docs)

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)
    return response




# ===============? this is for backend/rag_graph.py ?===============
# import os
# from dotenv import load_dotenv

# from langchain_community.vectorstores import FAISS
# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
# except ImportError:
#     from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # -------------------------------------------------
# # Load environment variables
# # -------------------------------------------------
# load_dotenv()

# # -------------------------------------------------
# # Paths
# # -------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "faiss_index")

# # -------------------------------------------------
# # Embeddings (must match ingest.py)
# # -------------------------------------------------
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # -------------------------------------------------
# # Load FAISS
# # -------------------------------------------------
# vectorstore = FAISS.load_local(
#     FAISS_INDEX_PATH,
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# # -------------------------------------------------
# # Gemini LLM
# # -------------------------------------------------
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-2.0-flash",
#     temperature=0.2
# )

# # -------------------------------------------------
# # Prompt
# # -------------------------------------------------
# prompt = ChatPromptTemplate.from_template(
#     """
# You are a helpful AI assistant.
# Answer the question ONLY using the context below.
# If the answer is not in the context, say "I don't know".

# Context:
# {context}

# Question:
# {question}
# """
# )

# chain = prompt | llm | StrOutputParser()

# # -------------------------------------------------
# # Helper
# # -------------------------------------------------
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # -------------------------------------------------
# # ✅ EXPORTED FUNCTION (FIXED)
# # -------------------------------------------------
# def rag_app(question: str) -> str:
#     # 🔴 OLD (removed in new LangChain)
#     # docs = retriever.get_relevant_documents(question)

#     # ✅ NEW (correct)
#     docs = retriever.invoke(question)

#     context = format_docs(docs)

#     response = chain.invoke({
#         "context": context,
#         "question": question
#     })

#     return response
