import logging
import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__) 



def clean_text(text):
    text = re.sub(r's+'," ",text.strip())
    text = re.sub(r'[^\w\s.,!?]','',text)
    return text

def load_documents(data_dir, file_types=[".pdf",".txt"]):
    try:
        loader = PyPDFLoader(data_dir)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {data_dir}")
        return documents
    except Exception as e:
       logger.error(f"Error loading documents: {str(e)}") 
       raise


def chunk_documents(documents, chunk_size=500, chunk_overlap=200):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    length_function=len,
                                                    add_start_index= True)
        chunks = text_splitter.split_documents(documents)
        for chunk in chunks:
            chunk.page_content = clean_text(chunk.page_content)
            chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}_{chunk.metadata.get('page', 0)}"
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {str(e)}")
        raise
    
    
def get_embeddings():
    try:
        import torch
        embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3',
                                        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                        encode_kwargs = {"normalize_embeddings": True})
        
        logger.info("Initialized bge-m3 embeddings")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}")
        raise
    
def process_documents(data_dir, output_dir):
    try:
        documents = load_documents(data_dir)
        if not documents:
            raise ValueError("No documents loaded")
        
        chunks = chunk_documents(documents)
        
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(output_dir, exist_ok=True)
        vectorstore.save_local(output_dir)
        logger.info(f"Saved FAISS index to {output_dir}")
        return len(chunks)
    
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        

if __name__ == "__main__":
    # Define the directory containing your source documents
    data_directory = r"C:\Users\rnmpr\Documents\rag-chatbot\src\mmlbook.pdf"
    
    # Define the directory where the FAISS index will be saved
    INDEX_OUTPUT_DIRECTORY = "models/embeddings/faiss_index_bge_m3"
    
    print("Starting document processing...")
    chunks_created = process_documents(
        data_dir=data_directory, 
        output_dir=INDEX_OUTPUT_DIRECTORY
    )
    if chunks_created:
        print(f"Successfully processed documents and created {chunks_created} chunks.")
        print(f"FAISS index saved to {INDEX_OUTPUT_DIRECTORY}")
    else:
        print("Document processing failed. Please check the logs.") #