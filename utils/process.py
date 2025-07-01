import os
import pickle
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Loaders
from langchain_community.document_loaders import UnstructuredURLLoader
try:
    from langchain_community.document_loaders import PlaywrightURLLoader
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    from langchain_community.document_loaders import PyPDFLoader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

def load_urls(url_list):
    print("LOADING URLS:", url_list)
    docs = []
    for url in url_list:
        ext = url.split("?")[0].split("#")[0].split(".")[-1].lower()

        if ext == "pdf" and HAS_PDF:
            try:
                print(f"🔄 Downloading PDF: {url}")
                pdf_path = "temp_downloaded.pdf"
                res = requests.get(url, timeout=10)
                res.raise_for_status()
                with open(pdf_path, "wb") as f:
                    f.write(res.content)
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                
                # Combine all PDF pages into single document
                if pdf_docs:
                    combined_content = "\n\n".join([doc.page_content for doc in pdf_docs])
                    combined_doc = Document(
                        page_content=combined_content,
                        metadata={"source": url, "type": "pdf", "pages": len(pdf_docs)}
                    )
                    docs.append(combined_doc)
                    print(f"✅ Loaded PDF with {len(pdf_docs)} pages: {url}")
                
                os.remove(pdf_path)
                continue
            except Exception as pe:
                print(f"❌ Failed to load PDF: {pe}")

        # Try UnstructuredURLLoader first
        try:
            print(f"🔄 Trying UnstructuredURLLoader for: {url}")
            loader = UnstructuredURLLoader(urls=[url])
            url_docs = loader.load()
            if url_docs and len(url_docs) > 0 and url_docs[0].page_content.strip():
                docs.extend(url_docs)
                print(f"✅ Loaded with UnstructuredURLLoader: {url}")
                continue
            else:
                raise Exception("Empty content from UnstructuredURLLoader")
        except Exception as e:
            print(f"❌ UnstructuredURLLoader failed for {url}: {e}")

        # Try PlaywrightURLLoader second
        if HAS_PLAYWRIGHT:
            try:
                print(f"🔄 Trying PlaywrightURLLoader for: {url}")
                loader = PlaywrightURLLoader(urls=[url])
                url_docs = loader.load()
                if url_docs and len(url_docs) > 0 and url_docs[0].page_content.strip():
                    docs.extend(url_docs)
                    print(f"✅ Loaded with PlaywrightURLLoader: {url}")
                    continue
                else:
                    raise Exception("Empty content from PlaywrightURLLoader")
            except Exception as pe:
                print(f"❌ PlaywrightURLLoader failed for {url}: {pe}")

        # Fallback using requests + BeautifulSoup
        try:
            print(f"🔄 Trying fallback with requests + BeautifulSoup: {url}")
            res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            content = soup.get_text(separator="\n", strip=True)
            if content:
                docs.append(Document(page_content=content, metadata={"source": url}))
                print(f"✅ Loaded with BeautifulSoup fallback: {url}")
            else:
                print("❌ BeautifulSoup content is empty")
        except Exception as be:
            print(f"❌ BeautifulSoup fallback failed: {be}")

    print(f"📊 Total documents loaded: {len(docs)}")
    return docs

def process_docs(docs):
    if not docs:
        raise ValueError("No documents loaded. Please check URLs.")
    valid_docs = [doc for doc in docs if doc.page_content.strip()]
    if not valid_docs:
        raise ValueError("All documents are empty. Please check URL content.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(valid_docs)
    print("TOTAL CHUNKS GENERATED:", len(chunks))
    if not chunks:
        raise ValueError("No chunks generated. Please check input content.")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_faiss_store():
    with open("faiss_store.pkl", "rb") as f:
        return pickle.load(f)

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
            },
            timeout=30
        )
        data = response.json()
        print("📦 Perplexity Response:", data)
        if 'error' in data:
            return f"⚠️ Perplexity API error: {data['error']['message']}"
        return data['choices'][0]['message']['content']
    except Exception as e:
        print("❌ Perplexity API error (summary):", e)
        return f"⚠️ Failed to fetch summary from Perplexity: {e}"

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
            },
            timeout=30
        )
        data = response.json()
        if 'error' in data:
            return f"⚠️ Perplexity API error: {data['error']['message']}"
        return data['choices'][0]['message']['content']
    except Exception as e:
        print("❌ Perplexity API error (Q&A):", e)
        return f"⚠️ Failed to fetch answer from Perplexity: {e}"