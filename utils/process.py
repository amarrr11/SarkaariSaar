import os
import requests
import pickle
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# ------------------- Load URL Content -------------------
def load_urls(url_list):
    print("LOADING URLS:", url_list)
    docs = []
    for url in url_list:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            main_content = soup.get_text(separator="\n", strip=True)
            doc = Document(page_content=main_content, metadata={"source": url})
            docs.append(doc)
        except Exception as e:
            print(f"❌ Error loading {url} -> {str(e)}")
    print("LOADED DOCS:", len(docs))
    return docs

# ------------------- Process & Embed -------------------
def process_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print("TOTAL CHUNKS GENERATED:", len(chunks))
    if len(chunks) == 0:
        raise ValueError("No chunks generated. Please check input content.")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("faiss_store")
    return vectorstore

# ------------------- Load FAISS DB -------------------
def load_faiss_store():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_store", embedding)

# ------------------- Summarize Using Perplexity API -------------------
def summarize_scheme(text):
    prompt = f"""
You are an expert in understanding government schemes. Summarize the following into:

1. Scheme Benefits  
2. Eligibility Criteria  
3. Documents Required  
4. Application Process

Text:
{text}
    """
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        result = response.json()
        print("📦 Perplexity Response:", result)
        return result['choices'][0]['message']['content']
    except Exception as e:
        print("❌ Perplexity API error:", e)
        return "⚠️ Failed to fetch summary from Perplexity."

# ------------------- Q&A Using Perplexity -------------------
def ask_question(question, context_chunks):
    context = "\n\n".join([doc.page_content for doc in context_chunks])
    prompt = f"""
You are an intelligent assistant. Using the context below, answer the following question:

Context:
{context}

Question:
{question}

Return the answer in a clear and helpful way.
    """
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print("❌ Perplexity API error:", e)
        return f"⚠️ Failed to fetch answer from Perplexity: {e}"
