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
import re

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
    """
    Loads PDF documents from the specified directory.

    Args:
        path (str): The directory path containing PDF files.

    Returns:
        List[Document]: A list of loaded PDF documents.
    """
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    return documents
    
def split_documents(documents):
    """
    Splits documents into smaller chunks for processing.

    Args:
        documents (List[Document]): A list of documents to be split.

    Returns:
        List[Document]: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(text_chunks, api_key):
    """
    Creates a vector store from text chunks using embeddings.

    Args:
        text_chunks (List[Document]): A list of text chunks.
        api_key (str): The API key for accessing embeddings.

    Returns:
        Chroma: A Chroma vector store.
    """
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = "./chroma_persist"
    
    vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory="./chroma_persist",  
                                         client_settings= Settings(anonymized_telemetry=False, is_persistent=True))
    return vector_store


class CustomBasicRagRetriever(BaseRetriever):
    """
    Custom retriever for basic RAG (Retrieval-Augmented Generation).

    Attributes:
        vector_store (any): The vector store for document retrieval.
    """
    vector_store: any

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents based on the query.

        Args:
            query (str): The query string.

        Returns:
            List[Document]: A list of relevant documents.
        """
        docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(query)
        docs = []
        for doc, score in docs_with_scores:
            doc.metadata["score"] = score
            docs.append(doc)
        
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronously retrieves relevant documents based on the query.

        Args:
            query (str): The query string.

        Returns:
            List[Document]: A list of relevant documents.
        """
        return self._get_relevant_documents(query)


def create_qa_chain_basic_rag(vector_store, api_key, llm):
    """
    Creates a QA chain using basic RAG retriever.

    Args:
        vector_store (Chroma): The vector store for document retrieval.
        api_key (str): The API key for accessing embeddings.
        llm (HuggingFaceHub): The language model.

    Returns:
        RetrievalQA: The QA chain.
    """
    basic_rag_retriever = CustomBasicRagRetriever(vector_store=vector_store)
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=basic_rag_retriever,
        return_source_documents=True
    )
    
    return chain


def create_qa_chain_rerank_cross_encoder(vector_store, api_key, llm):
    """
    Creates a QA chain using RAG retriever with cross encoder reranking.

    Args:
        vector_store (Chroma): The vector store for document retrieval.
        api_key (str): The API key for accessing embeddings.
        llm (HuggingFaceHub): The language model.

    Returns:
        RetrievalQA: The QA chain.
    """
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
    """
    Fetches the ground truth answer for a given question.

    Args:
        question (str): The question string.

    Returns:
        str: The ground truth answer.
    """
    ground_truth_data = {
        "What was Qualcomm's revenue in Q1 2023?": "Qualcomm's revenue in Q1 2023 was $9.46 billion.",
        "What were Qualcomm's earnings per share in Q2 2023?": "Qualcomm's earnings per share in Q2 2023 were $2.15.",
        "What was Qualcomm's net income in Q3 2023?": "Qualcomm's net income in Q3 2023 was $1.8 billion.",
        #"What Qualcomm's Q1 Revenue in 2025" : "Qualcomm's Q1 Revenue in 2025 was $11.669 billio."
    }
    return ground_truth_data.get(question, "Ground truth not available.")


def parse_evaluation_results(qa_eval):
    """
    Parses the QA evaluation text and calculates correct/incorrect answers.

    Args:
        qa_eval (dict or list): The QA evaluation results.

    Returns:
        tuple: Total questions, correct answers, incorrect answers, correct percentage, incorrect percentage.
    """
    if isinstance(qa_eval, dict):  
        qa_eval = [qa_eval]  # Convert single dictionary to a list

    total_questions = 0
    correct_answers = 0
    incorrect_answers = 0

    for res in qa_eval:
        reasoning_text = res.get("reasoning", "")
        matches = re.findall(r"GRADE: (CORRECT|INCORRECT)", reasoning_text)
        
        total_questions += len(matches)
        correct_answers += matches.count("CORRECT")
        incorrect_answers += matches.count("INCORRECT")
    
    correct_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    incorrect_percentage = (incorrect_answers / total_questions) * 100 if total_questions > 0 else 0
    
    return total_questions, correct_answers, incorrect_answers, correct_percentage, incorrect_percentage

def evaluate_qa(query, predicted_answer, ground_truth, llm):
    """
    Evaluates the QA performance using the given LLM.

    Args:
        query (str): The query string.
        predicted_answer (str): The predicted answer.
        ground_truth (str): The ground truth answer.
        llm (HuggingFaceHub): The language model.

    Returns:
        dict: The QA evaluation results.
    """
    qa_evaluator = load_evaluator(EvaluatorType.QA, llm=llm)
    qa_results = qa_evaluator.evaluate_strings(
        prediction=predicted_answer,
        reference=ground_truth,
        input=query
    )
    
    #st.write("QA Evaluation Results: ", qa_results)
    return qa_results


def validate_output(response):
    """
    Validates the output to filter hallucinated/misleading answers.

    Args:
        response (dict): The response dictionary.

    Returns:
        dict: The validated response.
    """
    hallucination_keywords = ["I think", "might be", "possibly", "uncertain", "estimate"]
    for keyword in hallucination_keywords:
        if keyword.lower() in response["result"].lower():
            response["result"] = "The response could not be validated. Please ask about Qualcomm's quarterly earnings."
            break
    return response


def compare_retrievers(user_question, api_key, vector_store):
    """
    Compares the performance of different retrievers.

    Args:
        user_question (str): The user's question.
        api_key (str): The API key for accessing embeddings.
        vector_store (Chroma): The vector store for document retrieval.
    """
    # Guardrail: Verify if the question is related to Qualcomm earnings
    guardrail_keywords = ["Qualcomm", "earnings", "quarterly", "financials", "revenue", "profit"]
    if not any(keyword.lower() in user_question.lower() for keyword in guardrail_keywords):
        st.warning("⚠️ I can only provide information about Qualcomm's quarterly earnings.")
        return
    
    # Set API key if provided
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        st.success("✅ Hugging Face API key set successfully!")
    
    # Initialize LLM (Mistral-7B)
    llm = HuggingFaceHub(
        #repo_id="HuggingFaceH4/zephyr-7b-beta",
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 1024},
    )
    
    # Retrieve ground truth
    ground_truth = get_ground_truth(user_question)

    # Run retrieval chains
    chain1 = create_qa_chain_basic_rag(vector_store, api_key, llm)
    chain2 = create_qa_chain_rerank_cross_encoder(vector_store, api_key, llm)

    response1 = chain1.invoke({"query": user_question})
    response2 = chain2.invoke({"query": user_question})

    # Apply output-side guardrails
    response1 = validate_output(response1)
    response2 = validate_output(response2)

    # Extract answers
    answer1 = extract_answer(response1["result"])
    answer2 = extract_answer(response2["result"])

    # Display answers side by side
    st.header("📊 Results Comparison")

    #col1, col2 = st.columns(2)

    col1, spacer, col2 = st.columns([1, 0.1, 1])  # Spacer column for separation


    with col1:
        with st.container():
            st.subheader("Basic RAG Retriever")
            st.success(f"**Answer:** {answer1}")
            doc1 = response1["source_documents"][0]
            similarity1 = doc1.metadata.get('score', 0)  # Default to 0 if missing
    
            # Similarity Score with Percentage
            st.write(f"🔍 **Similarity Score:** `{similarity1 * 100:.1f}%`")
            st.progress(similarity1)  
    
            # Document Snippet with Consistent Formatting
            st.write("📄 **Document Snippet:**")
            st.markdown(
                f"<div style='border:1px solid #ddd; padding:10px; min-height:100px;'>{doc1.page_content[:200]}...</div>", 
                unsafe_allow_html=True
            )
    
            # Padding for alignment
            st.write("")
    
    with col2:
        with st.container():
            st.subheader("Rerank with Cross Encoder")
            st.success(f"**Answer:** {answer2}")
            doc2 = response2["source_documents"][0]
            similarity2 = doc2.metadata.get('score', 0)  # Default to 0 if missing
    
            # Similarity Score with Percentage
            st.write(f"🔍 **Similarity Score:** `{similarity2 * 100:.1f}%`")
            st.progress(similarity2)  
    
            # Document Snippet with Consistent Formatting
            st.write("📄 **Document Snippet:**")
            st.markdown(
                f"<div style='border:1px solid #ddd; padding:10px; min-height:100px;'>{doc2.page_content[:200]}...</div>", 
                unsafe_allow_html=True
            )
    
            # Padding for alignment
            st.write("")

    # QA Evaluation
    st.header("LLM based QA Evaluation")
    qa_eval1 = evaluate_qa(user_question, answer1, ground_truth, llm)
    qa_eval2 = evaluate_qa(user_question, answer2, ground_truth, llm)

    # Process QA evaluations separately for both retrievers
    total_q1, correct_q1, incorrect_q1, correct_pct1, incorrect_pct1 = parse_evaluation_results(qa_eval1)
    total_q2, correct_q2, incorrect_q2, correct_pct2, incorrect_pct2 = parse_evaluation_results(qa_eval2)

    # Display results for Basic RAG Retriever
    st.subheader("🔹 Basic RAG Retriever Evaluation")
    st.write(f"**Total Questions:** {total_q1}")
    st.write(f"✅ **Correct Answers:** {correct_q1} ({correct_pct1:.1f}%)")
    st.write(f"❌ **Incorrect Answers:** {incorrect_q1} ({incorrect_pct1:.1f}%)")
    st.progress(correct_pct1 / 100)

    # Display results for Rerank with Cross Encoder
    st.subheader("🔹 Rerank with Cross Encoder Evaluation")
    st.write(f"**Total Questions:** {total_q2}")
    st.write(f"✅ **Correct Answers:** {correct_q2} ({correct_pct2:.1f}%)")
    st.write(f"❌ **Incorrect Answers:** {incorrect_q2} ({incorrect_pct2:.1f}%)")
    st.progress(correct_pct2 / 100)

# Helper function to extract answer from response
def extract_answer(response_text):
    """
    Extracts the answer from the response text.

    Args:
        response_text (str): The response text.

    Returns:
        str: The extracted answer.
    """
    answer_start = response_text.find("Answer:") + len("Answer:")
    return response_text[answer_start:].strip()

def main():
    """
    The main function to run the Streamlit app.
    """
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
