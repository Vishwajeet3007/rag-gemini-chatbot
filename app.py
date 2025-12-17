import streamlit as st
from backend.rag_graph import rag_app

st.set_page_config(page_title="Gemini RAG Chatbot", layout="centered")

st.title("🤖 Gemini RAG Chatbot")

query = st.text_input("Ask a question from the document:")

if query:
    with st.spinner("Thinking..."):
        answer = rag_app(query)
    st.markdown("### Answer")
    st.write(answer)
