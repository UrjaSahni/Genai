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
.token-status {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
.token-success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.token-error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# ---------------- OPENROUTER CLIENT SETUP ----------------
@st.cache_resource
def setup_openrouter_client():
    """Initialize OpenRouter client with error handling"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Check if API key exists
    if not api_key:
        st.error("❌ OPENROUTER_API_KEY not found in environment variables!")
        st.info("Please set your OpenRouter API key in the .env file")
        return None
    
    # Validate API key format
    if not api_key.startswith('sk-or-'):
        st.error("❌ Invalid API key format. OpenRouter API keys should start with 'sk-or-'")
        return None
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Test the connection with a simple request
        test_response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/your-repo", # Optional
                "X-Title": "Research Paper Summarizer", # Optional
            },
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        st.success("✅ OpenRouter client initialized successfully!")
        return client
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            st.error("❌ Authentication failed: Invalid OpenRouter API key")
            st.info("""
            **How to fix this:**
            1. Go to https://openrouter.ai/keys
            2. Create a new API key
            3. Update your .env file with: OPENROUTER_API_KEY=your_api_key_here
            4. Restart the application
            """)
        else:
            st.error(f"❌ Failed to initialize OpenRouter client: {error_msg}")
        return None

# Initialize client
openrouter_client = setup_openrouter_client()

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
        self.model = "deepseek/deepseek-chat-v3.1:free"
    
    def generate_summary(self, content: str, summary_type: str = "comprehensive") -> str:
        """Generate summary using OpenRouter"""
        if not self.client:
            return "❌ Error: AI client not available. Please check your OpenRouter API key."
            
        prompts = {
            "comprehensive": f"""
            Please provide a comprehensive summary of this research paper including:
            1. Main objective and research question
            2. Methodology used
            3. Key findings and results
            4. Conclusions and implications
            5. Limitations and future work
            
            Research Paper Content:
            {content[:4000]}
            """,
            "key_insights": f"""
            Extract the most important insights and findings from this research paper.
            Focus on novel contributions, significant results, and practical implications.
            Present the insights in a clear, structured format.
            
            Research Paper Content:
            {content[:4000]}
            """,
            "methodology": f"""
            Summarize the research methodology, experimental design, and analytical approaches used in this paper.
            Include details about data collection, analysis methods, and experimental setup.
            
            Research Paper Content:
            {content[:4000]}
            """
        }
        
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/research-summarizer",
                    "X-Title": "AI Research Paper Summarizer",
                },
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompts.get(summary_type, prompts["comprehensive"])
                }],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                return "❌ Authentication Error: Please check your OpenRouter API key."
            elif "429" in error_msg:
                return "❌ Rate Limit Error: Too many requests. Please wait and try again."
            elif "503" in error_msg:
                return "❌ Service Unavailable: The model is currently unavailable. Please try again later."
            elif "400" in error_msg:
                return "❌ Bad Request: The request was invalid. Please try with a different document."
            else:
                return f"❌ Error generating summary: {error_msg}"
    
    def compare_papers(self, summaries: List[str], titles: List[str]) -> str:
        """Compare multiple research papers"""
        if not self.client:
            return "❌ Error: AI client not available. Please check your OpenRouter API key."
            
        comparison_prompt = f"""
        Compare and analyze the following research papers. Provide a detailed analysis including:
        
        1. **Common Themes and Areas of Agreement:**
           - Shared research objectives
           - Similar methodological approaches
           - Consistent findings across papers
        
        2. **Contradictory Findings or Approaches:**
           - Conflicting results
           - Different methodological choices
           - Opposing conclusions
        
        3. **Research Gaps and Unexplored Areas:**
           - Topics not covered by any paper
           - Limitations mentioned across studies
           - Future research opportunities
        
        4. **Methodological Differences:**
           - Different experimental designs
           - Varied data collection methods
           - Alternative analytical approaches
        
        5. **Overall Synthesis and Key Insights:**
           - What can we learn from these papers collectively?
           - How do they contribute to the field?
           - Recommendations for future research
        
        Papers to analyze:
        """
        
        for i, (title, summary) in enumerate(zip(titles, summaries), 1):
            comparison_prompt += f"\n**Paper {i}: {title}**\n{summary[:1500]}\n"
        
        try:
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/research-summarizer",
                    "X-Title": "AI Research Paper Comparator",
                },
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": comparison_prompt
                }],
                max_tokens=3000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                return "❌ Authentication Error: Please check your OpenRouter API key."
            else:
                return f"❌ Error comparing papers: {error_msg}"

# ---------------- MAIN APPLICATION ----------------
def main():
    # Header
    st.markdown('<div class="main-header">🔬 AI Research Paper Summarizer & Comparator</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload research papers and get AI-powered summaries and comparative analysis</div>', 
                unsafe_allow_html=True)
    
    # Initialize processors
    doc_processor = SimpleDocumentProcessor()
    summarizer = ResearchSummarizer(openrouter_client)
    
    # Initialize session state
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📚 Upload Research Papers")
        
        # Show API key status
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key and api_key.startswith('sk-or-'):
            st.markdown('<div class="token-status token-success">✅ OpenRouter API Key: Configured</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="token-status token-error">❌ OpenRouter API Key: Not configured</div>', 
                       unsafe_allow_html=True)
            st.markdown("""
            **Setup Required:**
            1. Go to [OpenRouter Keys](https://openrouter.ai/keys)
            2. Create a new API key
            3. Add to .env file: `OPENROUTER_API_KEY=your_key_here`
            4. Restart the application
            """)
        
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
        
        # Model info
        st.markdown("## 🤖 AI Model Info")
        st.info("**Model:** DeepSeek Chat V3.1 (Free)\n**Provider:** OpenRouter")
        
        # Clear papers button
        if st.button("🗑️ Clear All Papers"):
            st.session_state.papers = []
            st.session_state.summaries = []
            st.rerun()
        
        # Instructions
        st.markdown("## 📝 Instructions")
        st.markdown("""
        1. **Setup**: Configure OpenRouter API key
        2. **Upload**: Add PDF or text files
        3. **Process**: Choose summary type
        4. **Analyze**: View individual summaries
        5. **Compare**: Generate comparative analysis
        6. **Export**: Download results as PDF/JSON
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
                            
                            # Generate summary if client is available
                            if openrouter_client:
                                with st.spinner("Generating AI summary..."):
                                    summary = summarizer.generate_summary(content, summary_type)
                                    st.session_state.summaries.append({
                                        'title': uploaded_file.name,
                                        'summary': summary,
                                        'type': summary_type
                                    })
                            else:
                                st.session_state.summaries.append({
                                    'title': uploaded_file.name,
                                    'summary': "❌ Summary not generated: API client not available",
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
        if st.session_state.papers:
            for i, paper in enumerate(st.session_state.papers):
                with st.expander(f"📑 {paper['name']}", expanded=False):
                    st.markdown(f"**Status:** {'✅ Processed' if paper['processed'] else '⏳ Processing...'}")
                    st.markdown(f"**File Size:** {len(paper['content'])} characters")
                    st.markdown(f"**Content Preview:**")
                    preview_text = paper['content'][:500] + "..." if len(paper['content']) > 500 else paper['content']
                    st.text_area("Content", preview_text, height=150, key=f"content_{i}", disabled=True)
        else:
            st.info("📤 No papers uploaded yet. Use the sidebar to upload research papers.")
    
    with col2:
        st.markdown("### 📊 Analysis & Summaries")
        
        if st.session_state.summaries:
            # Display individual summaries
            for i, summary in enumerate(st.session_state.summaries):
                with st.expander(f"📄 {summary['title']}", expanded=True):
                    st.markdown(f"**Summary Type:** {summary['type'].title()}")
                    if summary['summary'].startswith('❌'):
                        st.error(summary['summary'])
                    else:
                        st.markdown(f'<div class="summary-box">{summary["summary"]}</div>', 
                                   unsafe_allow_html=True)
        else:
            st.info("📋 No summaries available yet. Upload papers to generate summaries.")
        
        # Comparison section
        if len(st.session_state.summaries) >= 2:
            st.markdown("---")
            st.markdown("### 🔄 Comparative Analysis")
            
            # Only show button if we have valid summaries
            valid_summaries = [s for s in st.session_state.summaries if not s['summary'].startswith('❌')]
            
            if len(valid_summaries) >= 2:
                if st.button("🔍 Generate Comparative Analysis", type="primary"):
                    with st.spinner("Generating comparative analysis..."):
                        summaries_text = [s['summary'] for s in valid_summaries]
                        titles = [s['title'] for s in valid_summaries]
                        
                        comparison = summarizer.compare_papers(summaries_text, titles)
                        
                        if comparison.startswith('❌'):
                            st.error(comparison)
                        else:
                            st.markdown(f'<div class="comparison-box">', unsafe_allow_html=True)
                            st.markdown("**🔄 Comparative Analysis Results:**")
                            st.markdown(comparison)
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Need at least 2 successfully processed papers for comparison")
        elif len(st.session_state.summaries) == 1:
            st.info("📝 Upload one more paper to enable comparative analysis")
    
    # Export section
    if st.session_state.summaries:
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
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
                    
                    # Add metadata
                    pdf.set_font("Arial", size=10)
                    pdf.cell(0, 8, f"Generated using DeepSeek Chat V3.1 via OpenRouter", ln=True)
                    pdf.cell(0, 8, f"Number of papers analyzed: {len(st.session_state.papers)}", ln=True)
                    pdf.ln(5)
                    
                    # Add summaries
                    for i, summary in enumerate(st.session_state.summaries, 1):
                        pdf.set_font("Arial", 'B', 14)
                        title = summary['title'].encode('latin1', 'ignore').decode('latin1')
                        pdf.cell(0, 10, f"Paper {i}: {title}", ln=True)
                        
                        pdf.set_font("Arial", 'I', 10)
                        pdf.cell(0, 8, f"Summary Type: {summary['type'].title()}", ln=True)
                        
                        pdf.set_font("Arial", size=10)
                        clean_summary = summary['summary'].encode('latin1', 'ignore').decode('latin1')
                        pdf.multi_cell(0, 6, clean_summary)
                        pdf.ln(8)
                    
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
                    'metadata': {
                        'ai_model': 'deepseek/deepseek-chat-v3.1:free',
                        'provider': 'OpenRouter',
                        'total_papers': len(st.session_state.papers)
                    },
                    'papers': [
                        {
                            'name': p['name'], 
                            'content_length': len(p['content']),
                            'content_preview': p['content'][:500] + "..." if len(p['content']) > 500 else p['content']
                        } 
                        for p in st.session_state.papers
                    ],
                    'summaries': st.session_state.summaries
                }
                
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    "⬇️ Download JSON Data",
                    data=json_str,
                    file_name="research_analysis_data.json",
                    mime="application/json"
                )
        
        with col_export3:
            if st.button("📋 Export Summary Text"):
                # Create plain text summary
                text_content = "RESEARCH PAPER ANALYSIS SUMMARY\n"
                text_content += "=" * 50 + "\n\n"
                
                for i, summary in enumerate(st.session_state.summaries, 1):
                    text_content += f"PAPER {i}: {summary['title']}\n"
                    text_content += f"Summary Type: {summary['type'].title()}\n"
                    text_content += "-" * 30 + "\n"
                    text_content += f"{summary['summary']}\n\n"
                
                st.download_button(
                    "⬇️ Download Text Summary",
                    data=text_content,
                    file_name="research_summary.txt",
                    mime="text/plain"
                )
    
    # Status and Statistics
    if st.session_state.papers:
        st.markdown("---")
        st.markdown("### 📈 Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Papers Uploaded", len(st.session_state.papers))
        
        with col_stat2:
            st.metric("Summaries Generated", len(st.session_state.summaries))
        
        with col_stat3:
            total_chars = sum(len(p['content']) for p in st.session_state.papers)
            st.metric("Total Content", f"{total_chars:,} chars")
        
        with col_stat4:
            successful_summaries = len([s for s in st.session_state.summaries if not s['summary'].startswith('❌')])
            st.metric("Success Rate", f"{(successful_summaries/len(st.session_state.summaries)*100):.0f}%" if st.session_state.summaries else "0%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 AI-Powered Research Paper Summarizer & Comparator</p>
        <p>Built with Streamlit | Powered by DeepSeek via OpenRouter</p>
        <p>Authors: Urja Sahni & Sahil Kumar Sahoo</p>
        <p><small>Model: deepseek/deepseek-chat-v3.1:free</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
