import torch
import logging
from src.data_preprocessing import load_documents, chunk_documents, get_embeddings
from langchain.vectorstores import FAISS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_retriever(index_dir, search_type="similarity", k=5, use_mmr=False):
    try:
        embeddings = get_embeddings()
        
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_des = True)
        logger.info(f"Loaded FAISS vector store from {index_dir}")
        
        search_kwargs = {"k": k}
        if use_mmr:
            search_kwargs["fetch_k"] = k*2
            search_type = 'mmr'
            logger.info("Using Maximum Marginal Relevance (MMR) for retrieval")
        else:
            logger.info("Using similarity search for retrieval")
            
            
        retriever = vectorstore.as_retriever(search_type = search_type, search_kwargs = search_kwargs)
        return retriever
    except Exception as e:
        logger.error(f"Error setting up retriever: {str(e)}")
        raise
    
def retrieve_documents(retriever, query):
    try:
        documents = retriever.get_relevant_documents(query)
        if not documents:
            logger.warning(f"No documents retrieved for query: {query}")
            return []
        logger.info(f"Retrieved {len(documents)} documents for query: {query}")
        return [{"content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None)} for doc in documents]
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise