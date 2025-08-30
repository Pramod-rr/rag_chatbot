from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Load Mistral-7B-Instruct model wrapped for LangChain."""
    pipe = pipeline(
        "text-generation",
        model="mistralai/Mixtral-7B-Instruct-v0.3",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        max_new_tokens=512
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_rag_chain(retriever, language="English"):
    """Set up RAG chain with custom prompt and language support."""
    llm = get_llm()
    prompt_template = """
Use the following context to answer the question in {language}.
If unsure, say "I couldn't find relevant information."

Context: {context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "language"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt.partial(language=language)}
    )
