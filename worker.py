import os
import torch
import logging
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Globals
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None


def init_llm():
    global llm_hub, embeddings

    logger.info("Initializing Hugging Face LLM and embeddings...")

    # Hugging Face API token from .env
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    # Base LLM
    base_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=600,
    )

    # Wrap as chat model
    llm_hub = ChatHuggingFace(llm=base_llm)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
    )

    logger.info("LLM and embeddings initialized successfully.")


def process_document(document_path):
    global conversation_retrieval_chain

    logger.info(f"Processing document: {document_path}")

    loader = PyPDFLoader(document_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No text found in PDF (possibly scanned PDF).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
    )

    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(texts, embedding=embeddings)

    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
        input_key="question",
    )

    logger.info("Document processed and RetrievalQA chain ready.")


def process_prompt(prompt):
    global chat_history, conversation_retrieval_chain

    if conversation_retrieval_chain is None:
        return "Please upload a PDF first."

    output = conversation_retrieval_chain.invoke(
        {"question": prompt, "chat_history": chat_history}
    )

    answer = output["result"]
    chat_history.append((prompt, answer))

    return answer.strip()


# Initialize on startup
init_llm()
