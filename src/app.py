import streamlit as st
from src.retriever import get_retriever
from src.llm import get_rag_chain

st.title("Multilingual RAG Chatbot with bge-m3")

# Load retriever
retriever = get_retriever("models/embeddings/faiss_index_bge_m3")
rag_chain = get_rag_chain(retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (any language)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.run({"question": prompt, "language": "English"})
        st.markdown(response)
    st.session_state.messages.append({"role Snyder": "assistant", "content": response})