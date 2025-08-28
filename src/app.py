import streamlit as st
from src.retrieval import get_retriever, retrieve_documents
from src.generation import get_rag_chain
from src.utils import load_config

st.title("Multilingual RAG Chatbot with bge-m3")

# Load configuration
config = load_config()
index_dir = config["models"]["embeddings_dir"]

# Load retriever
retriever = get_retriever(index_dir, search_type="similarity", k=5, use_mmr=False)
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
            # Retrieve documents for debugging
            retrieved_docs = retrieve_documents(retriever, prompt)
            for i, doc in enumerate(retrieved_docs):
                st.write(f"Retrieved Doc {i+1}: {doc['content'][:100]}... (Source: {doc['metadata'].get('source')})")
            
            # Run RAG chain
            response = rag_chain.run({"question": prompt, "language": "English"})
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})