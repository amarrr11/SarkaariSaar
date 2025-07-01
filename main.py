import streamlit as st
import os
from utils.process import load_urls, process_docs, load_faiss_store, summarize_scheme, ask_question

st.set_page_config(page_title="SarkaariSaar - Scheme Research", layout="wide")
st.title("🗂️ Government Scheme Research Assistant")

# ---- SIDEBAR ----
st.sidebar.header("🔗 Load Scheme URLs")
url_input = st.sidebar.text_area("Paste scheme article URLs (one per line):")
process_btn = st.sidebar.button("⚙️ Process URLs")

# ---- MAIN WORK ----
if process_btn and url_input:
    urls = [u.strip() for u in url_input.split("\n") if u.strip()]
    st.info("🔄 Loading and processing content...")
    docs = load_urls(urls)

    if len(docs) == 0:
        st.error("❌ No documents could be loaded from the URLs.")
    else:
        st.success(f"✅ Successfully loaded {len(docs)} document(s).")
        try:
            process_docs(docs)
            st.success("✅ Content processed and saved (HuggingFace + FAISS index).")

            for doc in docs:
                st.subheader(f"🔗 {doc.metadata.get('source', 'Source URL')}")
                summary = summarize_scheme(doc.page_content)
                st.text_area("📄 Summary", summary, height=300)
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

# ---- Q&A ----
st.header("❓ Ask a question from the processed articles")
user_question = st.text_input("Your Question:")

faiss_store_exists = os.path.exists("faiss_store.pkl")

if user_question:
    if not faiss_store_exists:
        st.error("❌ Please process some scheme URLs first to enable Q&A.")
    else:
        try:
            vectorstore = load_faiss_store()
            docs = vectorstore.similarity_search(user_question, k=4)
            answer = ask_question(user_question, docs)

            st.subheader("🧠 Answer")
            st.write(answer)

            with st.expander("📚 Relevant Context Used"):
                for i, d in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(d.page_content)
        except Exception as e:
            st.error(f"❌ Error during Q&A: {e}")
