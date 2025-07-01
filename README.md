# 🗂️ SarkaariSaar - Government Scheme Research Assistant

SarkaariSaar is an intelligent research assistant that helps you understand government schemes better. Just provide a set of scheme-related URLs or PDFs, and the app will summarize each into four key points: **Benefits**, **Eligibility**, **Documents Required**, and **Application Process**. You can also ask questions and get context-based answers using AI-powered Q\&A.

---

## ✨ Features

* 🔗 **Multi-format Input**: Supports website URLs and PDF documents
* 🧠 **Smart Summarization**: Extracts core points like benefits, eligibility, documents needed, and how to apply
* ❓ **Interactive Q\&A**: Ask questions and receive AI-generated answers based on processed documents
* 📚 **Semantic Search**: Uses FAISS vector store with HuggingFace embeddings for fast and relevant document search
* 🔄 **Resilient Loaders**: Uses multiple loading strategies to ensure content gets extracted properly (Unstructured, Playwright, BeautifulSoup)

---

## 🚀 Quick Start

### ✅ Prerequisites

* Python 3.8+
* Perplexity API Key (free or paid)

---

### 🛠 Installation Steps

1. **Clone this repository**

   ```bash
   git clone https://github.com/amarrr11/SarkaariSaar.git
   cd SarkaariSaar
   ```

2. **Install all required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:

   ```env
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run main.py
   ```

---

## 🧩 How to Use

### 🔗 Process Documents

1. Paste one or more scheme article URLs in the sidebar
2. Click **"⚙️ Process URLs"**
3. View structured summaries for each document

### ❓ Ask Questions

1. After processing, scroll down to the Q\&A section
2. Enter your question — e.g., *“Who is eligible for this scheme?”*
3. Get contextual answers based on your uploaded content

---

## 📂 Supported Formats

* **Web Pages** (using URL)
* **PDFs** (multi-page, automatically merged)

---

## 🧠 Architecture Overview

| Component                  | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| **LangChain**              | For document loading, splitting, and managing pipelines |
| **HuggingFace Embeddings** | For generating sentence-level vector embeddings         |
| **FAISS**                  | Vector similarity search for relevant document chunks   |
| **Perplexity API**         | Powers intelligent summaries and Q\&A                   |
| **Streamlit**              | Frontend for seamless interaction                       |

---

## 🧱 Document Loading Strategy

1. **PDF Files** — Direct download and parsing using `PyPDFLoader`
2. **UnstructuredURLLoader** — Ideal for clean content extraction
3. **PlaywrightURLLoader** — For JS-heavy websites (if Playwright is installed)
4. **Fallback** — Uses BeautifulSoup as last resort for HTML scraping

---

## 🔐 Environment Variables

| Variable             | Purpose                | Required |
| -------------------- | ---------------------- | -------- |
| `PERPLEXITY_API_KEY` | API key for Perplexity | ✅ Yes    |

---

## 📦 Tech Stack

* **streamlit** - Web app interface
* **langchain** - Core processing framework
* **sentence-transformers** - MiniLM embeddings
* **faiss-cpu** - Vector search
* **python-dotenv** - Environment variable loader
* **requests**, **bs4** - Fallback web scraping tools

---

## 🛠 Troubleshooting

### ❌ Empty Content

* Double check your URLs are live and accessible
* Avoid login-restricted or JS-heavy pages (or use Playwright fallback)

### ❌ Perplexity API Error

* Check if API key is valid or rate-limited
* Review `.env` setup

### ❌ PDF Loading Error

* Ensure PDF links are direct file links
* Avoid protected or corrupted PDFs

---

## 💡 Future Enhancements (Suggestions)

* ✅ Local file uploader
* 🌍 Language translation support
* 🧩 Embedding visualization dashboard
* 📈 Scheme popularity analysis

---
