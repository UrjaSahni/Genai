import os
import tempfile
import streamlit as st
from pptx import Presentation
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether

# ---------------- SETUP ----------------

st.set_page_config(page_title="Research Paper Summarizer & Comparator", layout="wide")

# ---------------- SIDEBAR ----------------

with st.sidebar:
    if os.path.exists("TIETlogo.png"):
        st.image("TIETlogo.png", width=120)
    else:
        st.markdown("## 🏫 Research Paper Summarizer")
    st.markdown("## 🤖 Interactive Academic Assistant")
    st.markdown("Upload multiple research papers to get multi-level summaries and compare key contributions.")

# ---------------- LOAD DEFAULT PDFS ----------------

@st.cache_resource(show_spinner="Loading default PDFs...")
def load_default_vectorstore():
    # No default docs loaded for this domain-specific app; rely on user uploads
    from langchain.docstore.document import Document
    dummy_doc = Document(page_content="Upload research papers to start summarizing and comparing.", metadata={})
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([dummy_doc])
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

try:
    vector_store = load_default_vectorstore()
except Exception as e:
    st.error(f"Error initializing vector store: {str(e)}")
    st.stop()

# ---------------- FILE UPLOAD AND LOADING ----------------

def load_file_to_docs(file_path, ext):
    try:
        if ext == "pdf":
            return PyPDFLoader(file_path).load()
        elif ext == "docx":
            return UnstructuredWordDocumentLoader(file_path).load()
        elif ext == "pptx":
            prs = Presentation(file_path)
            text = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
            temp_txt = file_path + ".txt"
            with open(temp_txt, "w", encoding="utf-8") as f:
                f.write(text)
            return TextLoader(temp_txt).load()
        elif ext == "txt":
            return TextLoader(file_path, encoding="utf-8").load()
        elif ext == "md":
            return UnstructuredMarkdownLoader(file_path).load()
        return []
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []

uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF, DOCX, PPTX, TXT, MD):",
    type=["pdf", "docx", "pptx", "txt", "md"],
    accept_multiple_files=True
)

new_docs = []
if uploaded_files:
    for f in uploaded_files:
        ext = f.name.split(".")[-1].lower()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            loaded_docs = load_file_to_docs(tmp_path, ext)
            if loaded_docs:
                new_docs.extend(loaded_docs)
                st.success(f"✅ Loaded {f.name}")
            else:
                st.warning(f"⚠️ No content extracted from {f.name}")
            try:
                os.unlink(tmp_path)
            except:
                pass
        except Exception as e:
            st.error(f"Error processing {f.name}: {str(e)}")

# ---------------- VECTOR STORE UPDATE ----------------

if new_docs:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        new_chunks = splitter.split_documents(new_docs)
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_vs = FAISS.from_documents(new_chunks, embed)
        vector_store.merge_from(new_vs)
        st.success(f"✅ Indexed {len(new_docs)} documents")
    except Exception as e:
        st.error(f"Error updating vector store: {str(e)}")

# ---------------- LLM + RETRIEVER ----------------

try:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        st.error("❌ TOGETHER_API_KEY not found. Set it in environment variables.")
        st.stop()
    llm = ChatTogether(
        model="deepseek-ai/DeepSeek-V3",
        temperature=0.2,
        together_api_key=together_api_key
    )
except Exception as e:
    st.error(f"❌ Failed to initialize LLM or retriever: {str(e)}")
    st.stop()

# ---------------- MULTI-LEVEL SUMMARIZATION ----------------

def generate_multi_level_summary(paper_text, llm):
    prompt = f"""
    You are an expert academic research assistant. Summarize this research paper at three levels:

    1. Executive Summary (3-4 sentences)
    2. Section-wise summary (Abstract, Introduction, Methods, Results, Conclusion)
    3. Key Contributions (Highest impact points)

    Paper Text:
    {paper_text}

    Provide your response as a numbered list.
    """
    response = llm.run(prompt)
    return response

# ---------------- CROSS-PAPER COMPARISON ----------------

def compare_papers_summaries(summaries, llm):
    prompt = f"""
    You are comparing these research paper summaries. Highlight key overlapping points, contradictions,
    agreements, and notable gaps or future research opportunities.

    Summaries:
    {summaries}

    Provide a comprehensive comparison report.
    """
    response = llm.run(prompt)
    return response

# ---------------- DISPLAY UI ----------------

if new_docs:
    st.header("Research Paper Summarizer & Comparator")

    paper_summaries = []

    for idx, doc in enumerate(new_docs):
        paper_text = doc.page_content
        summary = generate_multi_level_summary(paper_text, llm)
        paper_summaries.append(f"Paper {idx+1}: {uploaded_files[idx].name}\n{summary}")

    st.subheader("Multi-level Summaries")
    for idx, summ in enumerate(paper_summaries):
        with st.expander(f"Summary for {uploaded_files[idx].name}"):
            st.write(summ)

    if len(paper_summaries) > 1:
        comparison = compare_papers_summaries("\n\n".join(paper_summaries), llm)
        st.subheader("Cross-Paper Comparison Report")
        st.write(comparison)

