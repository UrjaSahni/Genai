import os
import tempfile
import streamlit as st
import io
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- SETUP ----------------
load_dotenv()

st.set_page_config(
    page_title="AI Research Paper Summarizer & Comparator 🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.sub-header {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
.paper-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #2E86AB;
    margin: 1rem 0;
}
.summary-box {
    background: #e8f4fd;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.comparison-box {
    background: #f0f8e8;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.insight-box {
    background: #fff3cd;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.stButton > button {
    background-color: #2E86AB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #1B5E7F;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HUGGING FACE CLIENT SETUP ----------------
@st.cache_resource
def setup_hf_client():
    """Initialize Hugging Face client with error handling"""
    hf_token = os.getenv("HF_TOKEN", "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")
    
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )
    return client

# Initialize client
try:
    hf_client = setup_hf_client()
except Exception as e:
    st.error(f"Failed to initialize Hugging Face client: {str(e)}")
    st.stop()

# ---------------- DOCUMENT PROCESSING ----------------
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load document based on file type"""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_type == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                return []
            
            return loader.load()
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return []
    
    def process_documents(self, documents: List[Document]) -> FAISS:
        """Process documents and create vector store"""
        chunks = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store

# ---------------- AI SUMMARIZER ----------------
class ResearchSummarizer:
    def __init__(self, client):
        self.client = client
        self.model = "openai/gpt-oss-20b:nebius"
    
    def generate_summary(self, content: str, summary_type: str = "comprehensive") -> str:
        """Generate summary using Hugging Face model"""
        prompts = {
            "comprehensive": f"""
            Please provide a comprehensive summary of this research paper including:
            1. Main objective and research question
            2. Methodology used
            3. Key findings and results
            4. Conclusions and implications
            5. Limitations and future work
            
            Content: {content[:4000]}
            """,
            "key_insights": f"""
            Extract the most important insights and findings from this research paper.
            Focus on novel contributions, significant results, and practical implications.
            
            Content: {content[:4000]}
            """,
            "methodology": f"""
            Summarize the research methodology, experimental design, and analytical approaches used in this paper.
            
            Content: {content[:4000]}
            """
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompts.get(summary_type, prompts["comprehensive"])
                }],
                max_tokens=1500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def compare_papers(self, summaries: List[str], titles: List[str]) -> str:
        """Compare multiple research papers"""
        comparison_prompt = f"""
        Compare and analyze the following research papers. Identify:
        1. Common themes and areas of agreement
        2. Contradictory findings or approaches
        3. Research gaps and unexplored areas
        4. Methodological differences
        5. Overall synthesis and insights
        
        Papers to compare:
        """
        
        for i, (title, summary) in enumerate(zip(titles, summaries), 1):
            comparison_prompt += f"\nPaper {i} - {title}:\n{summary[:1500]}\n"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": comparison_prompt
                }],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error comparing papers: {str(e)}"

# ---------------- SIMILARITY ANALYZER ----------------
class SimilarityAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def calculate_similarity_matrix(self, texts: List[str], titles: List[str]) -> Dict:
        """Calculate similarity matrix between documents"""
        embeddings = self.model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        return {
            'matrix': similarity_matrix,
            'titles': titles,
            'embeddings': embeddings
        }
    
    def create_similarity_visualization(self, similarity_data: Dict):
        """Create similarity heatmap visualization"""
        fig = go.Figure(data=go.Heatmap(
            z=similarity_data['matrix'],
            x=similarity_data['titles'],
            y=similarity_data['titles'],
            colorscale='Blues',
            text=np.round(similarity_data['matrix'], 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Research Paper Similarity Matrix",
            xaxis_title="Papers",
            yaxis_title="Papers",
            font=dict(size=12)
        )
        
        return fig

# ---------------- MAIN APPLICATION ----------------
def main():
    # Header
    st.markdown('<div class="main-header">🔬 AI Research Paper Summarizer & Comparator</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload research papers and get AI-powered summaries and comparative analysis</div>', 
                unsafe_allow_html=True)
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    summarizer = ResearchSummarizer(hf_client)
    similarity_analyzer = SimilarityAnalyzer()
    
    # Initialize session state
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📚 Upload Research Papers")
        uploaded_files = st.file_uploader(
            "Select research papers",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDF, Word documents, or text files containing research papers"
        )
        
        # Processing options
        st.markdown("## ⚙️ Processing Options")
        summary_type = st.selectbox(
            "Summary Type",
            ["comprehensive", "key_insights", "methodology"],
            help="Choose the type of summary to generate"
        )
        
        # Clear papers button
        if st.button("🗑️ Clear All Papers"):
            st.session_state.papers = []
            st.session_state.summaries = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Uploaded Papers")
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if paper already exists
                if not any(paper['name'] == uploaded_file.name for paper in st.session_state.papers):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        # Load and process document
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        documents = doc_processor.load_document(tmp_path, file_type)
                        
                        if documents:
                            # Extract content for summarization
                            content = "\n\n".join([doc.page_content for doc in documents])
                            
                            # Store paper info
                            paper_info = {
                                'name': uploaded_file.name,
                                'content': content,
                                'documents': documents,
                                'processed': True
                            }
                            st.session_state.papers.append(paper_info)
                            
                            # Generate summary
                            summary = summarizer.generate_summary(content, summary_type)
                            st.session_state.summaries.append({
                                'title': uploaded_file.name,
                                'summary': summary,
                                'type': summary_type
                            })
                        
                        # Clean up
                        os.unlink(tmp_path)
        
        # Display uploaded papers
        for i, paper in enumerate(st.session_state.papers):
            with st.expander(f"📑 {paper['name']}", expanded=False):
                st.markdown(f"**Status:** {'✅ Processed' if paper['processed'] else '⏳ Processing...'}")
                st.markdown(f"**Content Preview:**")
                st.text(paper['content'][:500] + "..." if len(paper['content']) > 500 else paper['content'])
    
    with col2:
        st.markdown("### 📊 Analysis & Summaries")
        
        if st.session_state.summaries:
            # Display individual summaries
            for summary in st.session_state.summaries:
                st.markdown(f'<div class="summary-box">', unsafe_allow_html=True)
                st.markdown(f"**📄 {summary['title']}**")
                st.markdown(f"*Summary Type: {summary['type'].title()}*")
                st.markdown(summary['summary'])
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparison section
        if len(st.session_state.summaries) >= 2:
            st.markdown("### 🔄 Comparative Analysis")
            
            if st.button("🔍 Generate Comparative Analysis", type="primary"):
                with st.spinner("Generating comparative analysis..."):
                    summaries_text = [s['summary'] for s in st.session_state.summaries]
                    titles = [s['title'] for s in st.session_state.summaries]
                    
                    comparison = summarizer.compare_papers(summaries_text, titles)
                    
                    st.markdown(f'<div class="comparison-box">', unsafe_allow_html=True)
                    st.markdown("**🔄 Comparative Analysis Results:**")
                    st.markdown(comparison)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Similarity analysis
            st.markdown("### 📈 Similarity Analysis")
            
            if st.button("📊 Generate Similarity Matrix"):
                with st.spinner("Calculating paper similarities..."):
                    contents = [paper['content'] for paper in st.session_state.papers]
                    titles = [paper['name'] for paper in st.session_state.papers]
                    
                    similarity_data = similarity_analyzer.calculate_similarity_matrix(contents, titles)
                    fig = similarity_analyzer.create_similarity_visualization(similarity_data)
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Export section
    if st.session_state.summaries:
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📄 Export as PDF"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Add title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(0, 10, "Research Paper Analysis Report", ln=True, align='C')
                    pdf.ln(10)
                    
                    # Add summaries
                    for i, summary in enumerate(st.session_state.summaries, 1):
                        pdf.set_font("Arial", 'B', 14)
                        title = summary['title'].encode('latin1', 'ignore').decode('latin1')
                        pdf.cell(0, 10, f"Paper {i}: {title}", ln=True)
                        
                        pdf.set_font("Arial", size=10)
                        clean_summary = summary['summary'].encode('latin1', 'ignore').decode('latin1')
                        pdf.multi_cell(0, 8, clean_summary)
                        pdf.ln(5)
                    
                    # Create download
                    pdf_output = pdf.output(dest='S')
                    if isinstance(pdf_output, str):
                        pdf_bytes = pdf_output.encode('latin1')
                    else:
                        pdf_bytes = pdf_output
                    
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name="research_analysis_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"PDF export error: {str(e)}")
        
        with col_export2:
            if st.button("📊 Export as JSON"):
                export_data = {
                    'papers': [{'name': p['name'], 'content_preview': p['content'][:1000]} for p in st.session_state.papers],
                    'summaries': st.session_state.summaries,
                    'analysis_timestamp': str(st.session_state.get('timestamp', 'N/A'))
                }
                
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    "⬇️ Download JSON Data",
                    data=json_str,
                    file_name="research_analysis_data.json",
                    mime="application/json"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 AI-Powered Research Paper Summarizer & Comparator</p>
        <p>Built with Streamlit | Powered by Hugging Face | Authors: Urja Sahni & Sahil Kumar Sahoo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
