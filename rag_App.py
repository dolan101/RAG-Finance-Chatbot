import streamlit as st
from PyPDF2 import PdfReader
import os
import platform
if platform.system() == "Linux":
    import sys
    __import__('pysqlite3')
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules["pysqlite3"]
import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader

from langchain.chains import RetrievalQA

from sentence_transformers import CrossEncoder

from langchain_core.retrievers import BaseRetriever
#from langchain.retrievers import RerankRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import pandas as pd

from langchain_core.runnables import chain
from typing import List
from langchain_core.documents import Document

from collections import defaultdict

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import list_models

#from langchain.prompts import PromptTemplate
#from langchain.schema import Document
#from langchain.embeddings import GoogleGenerativeAIEmbeddings

#from langchain import load_stuff_docs_chain  # Ensure the correct import based on new guides
#from langchain.globals import set_verbose, get_verbose  
#
#st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Financial AI Assistant: Get Qualcomm's quarterly earnings
### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need API key for the chatbot to access Huggingface's Generative AI models. Obtain your Access Token https://huggingface.co/settings/tokens.

3. **Ask a Question**: Ask any questions related to Qualcomm's earnings from 2023 to 2025.
""")



# This is the first API key input; no need to repeat it in the main function.
api_key = st.text_input("Enter your Access Token:", type="password", key="api_key_input")
    
   
def pdf_loader(path = "/path/") :
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    return documents
    
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(text_chunks, api_key):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = "./chroma_persist"
    
    vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_persist",  
                                         client_settings= Settings(anonymized_telemetry=False, is_persistent=True))
    return vector_store


class CustomBasicRagRetriever(BaseRetriever):
    vector_store: any

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(query)
        docs = []
        for doc, score in docs_with_scores:
            doc.metadata["score"] = score
            docs.append(doc)
        
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def create_qa_chain_basic_rag(vector_store, api_key, llm):
    basic_rag_retriever = CustomBasicRagRetriever(vector_store=vector_store)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=basic_rag_retriever,
        return_source_documents=True
    )
    
    return chain


def create_qa_chain_rerank_cross_encoder(vector_store, api_key, llm):
    
    basic_rag_retriever = CustomBasicRagRetriever(vector_store=vector_store)
    
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    cross_encoder_compressor = CrossEncoderReranker(model=model, top_n=3)
    cross_encoder_retriever = ContextualCompressionRetriever(
        base_compressor=cross_encoder_compressor,
        base_retriever=basic_rag_retriever
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=cross_encoder_retriever,
        return_source_documents=True)
     
    return chain

def get_ground_truth(question):
    """Fetches the ground truth answer for a given question."""
    ground_truth_data = {
        "What was Qualcomm's revenue in Q1 2023?": "Qualcomm's revenue in Q1 2023 was $9.46 billion.",
        "What were Qualcomm's earnings per share in Q2 2023?": "Qualcomm's earnings per share in Q2 2023 were $2.15.",
        "What was Qualcomm's net income in Q3 2023?": "Qualcomm's net income in Q3 2023 was $1.8 billion."
    }
    return ground_truth_data.get(question, "Ground truth not available.")

def evaluate_qa(query, predicted_answer, ground_truth, llm):
    qa_evaluator = load_evaluator(EvaluatorType.QA, llm=llm)
    qa_results = qa_evaluator.evaluate_strings(
        prediction=predicted_answer,
        reference=ground_truth,
        input=query
    )
    
    st.write("QA Evaluation Results: ", qa_results)
    return qa_results


def validate_output(response):
    # Output-side guardrail to filter hallucinated/misleading answers
    hallucination_keywords = ["I think", "might be", "possibly", "uncertain", "estimate"]
    for keyword in hallucination_keywords:
        if keyword.lower() in response["result"].lower():
            response["result"] = "The response could not be validated. Please ask about Qualcomm's quarterly earnings."
            break
    return response

def compare_retrievers(user_question, api_key, vector_store):
    # Guardrail: Verify if the question is related to Qualcomm earnings
    guardrail_keywords = ["Qualcomm", "earnings", "quarterly", "financials", "revenue", "profit"]
    if not any(keyword.lower() in user_question.lower() for keyword in guardrail_keywords):
        st.write("Reply: I can only provide information about Qualcomm's quarterly earnings.")
        return
    
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    #llm_hf = HuggingFaceEndpoint(
    #    repo_id="HuggingFaceH4/zephyr-7b-beta",
    #    #repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #    #repo_id="Intel/dynamic_tinybert",
    #    #repo_id="HuggingFaceH4/zephyr-7b-beta",
    #    #repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #    task="text-generation",
    #    max_new_tokens=1024,
    #    do_sample=False,
    #    repetition_penalty=1.03,
    #)
    #llm = ChatHuggingFace(llm=llm_hf)
    
    #llm = ChatGoogleGenerativeAI(
    #    model="gemini-2.0-pro-exp-02-05", 
    #    temperature=0.9, 
    #    google_api_key=api_key
    #)
    
    
    
    llm = HuggingFaceHub(
        #repo_id="HuggingFaceH4/zephyr-7b-beta",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.3, "max_new_tokens":1024},
    )
    
    st.success("Hugging Face API key set successfully!")
    
    ground_truth = get_ground_truth(user_question)
    chain1 = create_qa_chain_basic_rag(vector_store, api_key, llm)
    chain2 = create_qa_chain_rerank_cross_encoder(vector_store, api_key, llm)

    response1 = chain1.invoke({"query": user_question})
    response2 = chain2.invoke({"query": user_question})

    st.write("## Results from BasicRagRetriever")
    # Apply output-side guardrail
    response1 = validate_output(response1)
    
    #st.write("**Reply:** ", response1)
    response_text = response1["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    st.write("**Answer:** ", answer)
    
    #st.write("**Reply:** ", response1["result"])
    ##for doc in response1["source_documents"][0]:
    doc = response1["source_documents"][0]
    st.write(f"**Document:** {doc.page_content[:200]}")
    st.write(f"**Similarity Score:** {doc.metadata.get('score', 'N/A')}")

    qa_eval = evaluate_qa(user_question, answer, ground_truth, llm)

    st.write("## Results from RerankWithCrossEncoder")
    response2 = validate_output(response2)
    
    #st.write("**Reply:** ", response2["result"])
    response_text = response2["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    st.write("**Answer:** ", answer)
    #for doc in response2["source_documents"][0]:
    doc = response2["source_documents"][0]
    st.write(f"**Document:** {doc.page_content[:200]}")
    st.write(f"**Similarity Score:** {doc.metadata.get('score', 'N/A')}")
    qa_eval = evaluate_qa(user_question, answer, ground_truth, llm)

def main():
    st.header("AI Financial Chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
        with st.spinner("Processing..."):
            raw_text = pdf_loader("./Qualcomm_earnings/")
            text_chunks = split_documents(raw_text)
            vector_store = get_vector_store(text_chunks, api_key)
            if user_question and api_key:  # Ensure API key and user question are provided
                compare_retrievers(user_question, api_key, vector_store)
            
            st.success("Done")

if __name__ == "__main__":
    main()
