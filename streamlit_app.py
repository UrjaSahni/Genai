import os
import tempfile
import streamlit as st
import io
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import PyPDF2
from fpdf import FPDF
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# ---------------- SETUP ----------------
load_dotenv()

st.set_page_config(
    page_title="AI Research Paper Summarizer & Comparator 🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (same as before)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #007acc;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .author-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HUGGING FACE MODEL SETUP ----------------
@st.cache_resource
def load_huggingface_model():
    """Load DeepSeek model from Hugging Face with caching"""
    try:
        model_name = "deepseek-ai/DeepSeek-V3.1-Base"
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            return None, "HF_TOKEN not found in environment variables!"
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        st.info(f"Loading model on {device}. This may take a few minutes...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        return pipe, f"DeepSeek-V3.1-Base loaded successfully on {device}!"
        
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"

# Initialize model
with st.spinner("Loading DeepSeek model... This may take a few minutes on first run."):
    text_generator, model_status = load_huggingface_model()

# Display model status
if text_generator is None:
    st.error(f"❌ {model_status}")
    st.info("💡 Make sure you have set your HF_TOKEN in the .env file")
    st.stop()
else:
    st.success(f"✅ {model_status}")

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
    def __init__(self, text_generator):
        self.text_generator = text_generator
    
    def generate_summary(self, content: str, summary_type: str = "comprehensive") -> str:
        """Generate summary using Hugging Face DeepSeek model"""
        if not self.text_generator:
            return "❌ Error: Text generator not available. Please check your model loading."
        
        # Truncate content to avoid token limits
        max_content_length = 2000  # Reduced for base model
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompts = {
            "comprehensive": f"""Please provide a comprehensive summary of this research paper including:
1. Main objective and research question
2. Methodology used
3. Key findings and results
4. Conclusions and implications
5. Limitations and future work

Research Paper Content:
{content}

Summary:""",
            
            "key_insights": f"""Extract the most important insights and findings from this research paper. Focus on:
- Novel contributions
- Significant results
- Practical implications
- Key takeaways

Research Paper Content:
{content}

Key Insights:""",
            
            "methodology": f"""Summarize the research methodology, experimental design, and analytical approaches used in this paper. Include details about:
- Data collection methods
- Analysis techniques
- Experimental setup
- Variables and measurements

Research Paper Content:
{content}

Methodology Summary:"""
        }
        
        try:
            prompt = prompts.get(summary_type, prompts["comprehensive"])
            
            # Generate response using the pipeline
            response = self.text_generator(
                prompt,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.text_generator.tokenizer.eos_token_id,
                eos_token_id=self.text_generator.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Extract generated text (remove the original prompt)
            generated_text = response[0]['generated_text']
            summary = generated_text.replace(prompt, "").strip()
            
            return summary if summary else "❌ Unable to generate summary. Please try with a different document."
            
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"
    
    def compare_papers(self, summaries: List[str], titles: List[str]) -> str:
        """Compare multiple research papers"""
        if not self.text_generator:
            return "❌ Error: Text generator not available."
        
        comparison_prompt = f"""Compare and analyze the following research papers. Provide a detailed analysis including:

1. Common Themes and Areas of Agreement
2. Contradictory Findings or Approaches
3. Research Gaps and Unexplored Areas
4. Methodological Differences
5. Overall Synthesis and Key Insights

Papers to analyze:
"""
        
        for i, (title, summary) in enumerate(zip(titles, summaries), 1):
            comparison_prompt += f"\nPaper {i}: {title}\n{summary[:800]}\n"
        
        comparison_prompt += "\nComparative Analysis:"
        
        try:
            response = self.text_generator(
                comparison_prompt,
                max_new_tokens=800,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.text_generator.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            comparison = generated_text.replace(comparison_prompt, "").strip()
            
            return comparison if comparison else "❌ Unable to generate comparison."
            
        except Exception as e:
            return f"❌ Error comparing papers: {str(e)}"

# ---------------- MAIN APPLICATION ----------------
def main():
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>🔬 AI-Powered Research Paper Summarizer & Comparator</h1>
        <p style="font-size: 18px; margin-bottom: 0;">Built with Streamlit | Powered by DeepSeek-V3.1-Base via Hugging Face</p>
    </div>
    
    <div class="author-info">
        <h3 style="margin: 0;">Authors: Urja Sahni & Sahil Kumar Sahoo</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Model: deepseek-ai/DeepSeek-V3.1-Base</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize components
    doc_processor = SimpleDocumentProcessor()
    ai_summarizer = ResearchSummarizer(text_generator)
    
    # Sidebar for features
    with st.sidebar:
        st.markdown("### 🎯 Key Features")
        
        st.markdown('''
        <div class="feature-box">
            <h4>📄 Document Processing</h4>
            <p>Support for PDF and text files with automatic text extraction</p>
        </div>
        
        <div class="feature-box">
            <h4>🤖 Local AI Processing</h4>
            <p>DeepSeek-V3.1-Base model running via Hugging Face Transformers</p>
        </div>
        
        <div class="feature-box">
            <h4>📊 Comparison Tool</h4>
            <p>Compare multiple research papers side-by-side</p>
        </div>
        
        <div class="feature-box">
            <h4>⚡ Multiple Summary Types</h4>
            <p>Comprehensive, Key Insights, and Methodology-focused summaries</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 System Requirements")
        st.markdown(f"""
        - **GPU Available**: {'✅ Yes' if torch.cuda.is_available() else '❌ No (CPU only)'}
        - **Model**: DeepSeek-V3.1-Base
        - **Memory**: {torch.cuda.get_device_properties(0).total_memory // 1024**3 if torch.cuda.is_available() else 'N/A'} GB VRAM
        """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📄 Single Paper Analysis", "📊 Multiple Papers Comparison", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Upload and Analyze a Single Research Paper")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a research paper (PDF or TXT)",
                type=['pdf', 'txt'],
                help="Upload a PDF or text file containing your research paper"
            )
        
        with col2:
            summary_type = st.selectbox(
                "Summary Type",
                ["comprehensive", "key_insights", "methodology"],
                format_func=lambda x: {
                    "comprehensive": "🔍 Comprehensive Summary",
                    "key_insights": "💡 Key Insights",
                    "methodology": "🔬 Methodology Focus"
                }[x]
            )
        
        if uploaded_file is not None:
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-container"><h4>{file_details["Filename"]}</h4><p>Filename</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-container"><h4>{file_details["File size"]}</h4><p>File Size</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-container"><h4>{file_details["File type"]}</h4><p>File Type</p></div>', unsafe_allow_html=True)
            
            # Process file
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Processing your research paper with DeepSeek model..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Extract text
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        extracted_text = doc_processor.process_file(tmp_file_path, file_extension)
                        
                        if extracted_text.strip():
                            st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
                            st.markdown(f"### 📋 Summary ({summary_type.replace('_', ' ').title()})")
                            
                            # Generate summary
                            with st.spinner("DeepSeek model is analyzing your paper..."):
                                summary = ai_summarizer.generate_summary(extracted_text, summary_type)
                            
                            st.markdown(summary)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show extracted text preview
                            with st.expander("📖 View Extracted Text (Preview)"):
                                preview_length = min(1000, len(extracted_text))
                                st.text_area(
                                    "Document Content",
                                    extracted_text[:preview_length] + ("..." if len(extracted_text) > preview_length else ""),
                                    height=200,
                                    disabled=True
                                )
                                st.info(f"Showing first {preview_length} characters of {len(extracted_text)} total characters")
                        else:
                            st.error("❌ Could not extract text from the uploaded file. Please check if the file is valid.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass
    
    with tab2:
        st.markdown("### Compare Multiple Research Papers")
        
        st.markdown('''
        <div class="feature-box">
            <h4>📊 Multi-Paper Analysis</h4>
            <p>Upload 2-3 research papers to compare methodologies, findings, and insights across studies.</p>
            <p><strong>Note:</strong> Due to model limitations, please use shorter papers for comparison.</p>
        </div>
        ''', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose research papers (PDF or TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload 2-3 research papers for comparison (smaller files recommended)"
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
            
            # Show files info
            st.markdown("### 📁 Uploaded Files")
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"**{i}.** {file.name} ({file.size/1024:.1f} KB)")
            
            if st.button("🔍 Compare Papers", type="primary", use_container_width=True):
                if len(uploaded_files) > 3:
                    st.warning("⚠️ Please upload maximum 3 files for optimal performance with the base model.")
                else:
                    summaries = []
                    titles = []
                    
                    with st.spinner("Processing and comparing papers with DeepSeek model..."):
                        progress_bar = st.progress(0)
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            # Update progress
                            progress = (i + 1) / len(uploaded_files)
                            progress_bar.progress(progress)
                            st.write(f"Processing {uploaded_file.name}...")
                            
                            # Save file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            try:
                                # Extract text and generate summary
                                file_extension = uploaded_file.name.split('.')[-1].lower()
                                extracted_text = doc_processor.process_file(tmp_file_path, file_extension)
                                
                                if extracted_text.strip():
                                    summary = ai_summarizer.generate_summary(extracted_text, "comprehensive")
                                    summaries.append(summary)
                                    titles.append(uploaded_file.name)
                                else:
                                    st.error(f"❌ Could not extract text from {uploaded_file.name}")
                            
                            except Exception as e:
                                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                            
                            finally:
                                try:
                                    os.unlink(tmp_file_path)
                                except:
                                    pass
                        
                        progress_bar.progress(1.0)
                    
                    if summaries and len(summaries) >= 2:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 Comparative Analysis")
                        
                        with st.spinner("Generating comparative analysis with DeepSeek..."):
                            comparison = ai_summarizer.compare_papers(summaries, titles)
                        
                        st.markdown(comparison)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show individual summaries
                        with st.expander("📄 Individual Paper Summaries"):
                            for title, summary in zip(titles, summaries):
                                st.markdown(f"#### {title}")
                                st.markdown(summary)
                                st.markdown("---")
                    else:
                        st.error("❌ Not enough valid papers processed for comparison. Please check your files.")
        
        elif uploaded_files and len(uploaded_files) == 1:
            st.info("ℹ️ Please upload at least 2 files for comparison.")
    
    with tab3:
        st.markdown("### ℹ️ About This Application")
        
        st.markdown('''
        <div class="feature-box">
            <h4>🎯 Purpose</h4>
            <p>This application helps researchers, students, and academics quickly summarize and compare research papers using the DeepSeek-V3.1-Base model via Hugging Face Transformers.</p>
        </div>
        
        <div class="feature-box">
            <h4>⚙️ Technical Details</h4>
            <ul>
                <li><strong>AI Model:</strong> DeepSeek-V3.1-Base (671B parameters, 37B active)</li>
                <li><strong>Provider:</strong> Hugging Face Transformers</li>
                <li><strong>Frontend:</strong> Streamlit</li>
                <li><strong>Document Processing:</strong> PyPDF2 for PDF extraction</li>
                <li><strong>File Support:</strong> PDF, TXT formats</li>
                <li><strong>Hardware:</strong> GPU acceleration when available</li>
            </ul>
        </div>
        
        <div class="feature-box">
            <h4>👥 Authors</h4>
            <p><strong>Urja Sahni</strong> & <strong>Sahil Kumar Sahoo</strong></p>
            <p>Final year students passionate about AI and research automation.</p>
        </div>
        
        <div class="feature-box">
            <h4>🔧 Setup Requirements</h4>
            <p>To run this application, you need:</p>
            <ul>
                <li>Python 3.8+</li>
                <li>Hugging Face token (for model access)</li>
                <li>GPU with sufficient VRAM (recommended: 16GB+)</li>
                <li>Required packages: streamlit, transformers, torch, PyPDF2, python-dotenv</li>
            </ul>
            <p><strong>Environment Variable:</strong> <code>HF_TOKEN</code></p>
        </div>
        ''', unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
