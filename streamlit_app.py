import os
import tempfile
import streamlit as st
import io
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
from fpdf import FPDF

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
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Hugging Face client: {str(e)}")
        return None

# Initialize client
hf_client = setup_hf_client()

# ---------------- DOCUMENT PROCESSING ----------------
class SimpleDocumentProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""
    
    def process_file(self, file_path: str, file_type: str) -> str:
        """Process file based on type"""
        if file_type == "pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_type in ["txt", "md"]:
            return self.extract_text_from_txt(file_path)
        else:
            return ""

# ---------------- AI SUMMARIZER ----------------
class ResearchSummarizer:
    def __init__(self, client):
        self.client = client
        self.model = "openai/gpt-oss-20b:nebius"
    
    def generate_summary(self, content: str, summary_type: str = "comprehensive") -> str:
        """Generate summary using Hugging Face model"""
        if not self.client:
            return "Error: AI client not available"
            
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
        if not self.client:
            return "Error: AI client not available"
            
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

# ---------------- MAIN APPLICATION ----------------
def main():
    # Header
    st.markdown('<div class="main-header">🔬 AI Research Paper Summarizer & Comparator</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload research papers and get AI-powered summaries and comparative analysis</div>', 
                unsafe_allow_html=True)
    
    # Initialize processors
    doc_processor = SimpleDocumentProcessor()
    summarizer = ResearchSummarizer(hf_client)
    
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
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDF or text files containing research papers"
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
        
        # Instructions
        st.markdown("## 📝 Instructions")
        st.markdown("""
        1. Upload PDF or text files
        2. Choose summary type
        3. Wait for processing
        4. View summaries and comparisons
        5. Export results
        """)
    
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
                        
                        # Extract text
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        content = doc_processor.process_file(tmp_path, file_type)
                        
                        if content.strip():
                            # Store paper info
                            paper_info = {
                                'name': uploaded_file.name,
                                'content': content,
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
                            
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                        else:
                            st.warning(f"⚠️ No text content found in {uploaded_file.name}")
                        
                        # Clean up
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
        
        # Display uploaded papers
        for i, paper in enumerate(st.session_state.papers):
            with st.expander(f"📑 {paper['name']}", expanded=False):
                st.markdown(f"**Status:** {'✅ Processed' if paper['processed'] else '⏳ Processing...'}")
                st.markdown(f"**Content Preview:**")
                preview_text = paper['content'][:500] + "..." if len(paper['content']) > 500 else paper['content']
                st.text_area("Content", preview_text, height=200, key=f"content_{i}")
    
    with col2:
        st.markdown("### 📊 Analysis & Summaries")
        
        if st.session_state.summaries:
            # Display individual summaries
            for i, summary in enumerate(st.session_state.summaries):
                with st.expander(f"📄 {summary['title']}", expanded=True):
                    st.markdown(f"**Summary Type:** {summary['type'].title()}")
                    st.markdown(f'<div class="summary-box">{summary["summary"]}</div>', 
                               unsafe_allow_html=True)
        
        # Comparison section
        if len(st.session_state.summaries) >= 2:
            st.markdown("---")
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
        else:
            st.info("Upload at least 2 papers to enable comparative analysis")
    
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
                    'timestamp': str(st.session_state.get('timestamp', 'N/A'))
                }
                
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    "⬇️ Download JSON Data",
                    data=json_str,
                    file_name="research_analysis_data.json",
                    mime="application/json"
                )
    
    # Status section
    if not st.session_state.papers:
        st.info("👆 Upload some research papers to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 AI-Powered Research Paper Summarizer & Comparator</p>
        <p>Built with Streamlit | Powered by Hugging Face</p>
        <p>Authors: Urja Sahni & Sahil Kumar Sahoo</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
