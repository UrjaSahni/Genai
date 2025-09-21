# AI-Powered Research Paper Summarizer & Comparator 🔬

A comprehensive tool for academic researchers to automatically summarize and compare research papers using advanced AI models through OpenRouter.

## 🌟 Features

- **Multi-Format Support**: Upload PDF, TXT, and Markdown files
- **AI-Powered Summarization**: Generate comprehensive summaries, key insights, or methodology overviews
- **Comparative Analysis**: Compare multiple papers to identify agreements, contradictions, and research gaps
- **Export Options**: Download results as PDF reports, JSON data, or plain text
- **Real-time Processing**: Instant feedback and progress indicators
- **Statistics Dashboard**: Track processing metrics and success rates

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd research-paper-summarizer

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. **Get OpenRouter API Key**:
   - Visit [OpenRouter Keys](https://openrouter.ai/keys)
   - Create a new API key
   - Copy the key (starts with `sk-or-`)

2. **Setup Environment**:
   - Copy the `.env` file template
   - Replace the API key with your actual key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 🛠️ Usage Guide

### Step 1: Upload Papers
- Use the sidebar to upload research papers (PDF, TXT, MD formats)
- Papers are processed automatically upon upload
- View upload status and content previews

### Step 2: Choose Summary Type
- **Comprehensive**: Complete overview including objectives, methodology, findings, and conclusions
- **Key Insights**: Focus on novel contributions and significant results
- **Methodology**: Detailed analysis of research methods and experimental design

### Step 3: Review Summaries
- Individual summaries appear in the Analysis & Summaries section
- Each summary is clearly labeled with the paper title and summary type

### Step 4: Generate Comparisons
- Upload 2+ papers to enable comparative analysis
- Click "Generate Comparative Analysis" for detailed comparison
- Results include common themes, contradictions, and research gaps

### Step 5: Export Results
- **PDF Report**: Formatted document with all summaries
- **JSON Data**: Structured data for further analysis
- **Text Summary**: Plain text format for easy sharing

## 🤖 AI Model Information

- **Model**: DeepSeek Chat V3.1 (Free)
- **Provider**: OpenRouter
- **Capabilities**: 
  - Advanced text comprehension
  - Academic writing analysis
  - Comparative reasoning
  - Structured output generation

## 📁 Project Structure

```
research-paper-summarizer/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create from template)
├── README.md                # This file
└── exports/                 # Generated exports (created automatically)
```

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web interface framework
- **OpenAI**: API client for OpenRouter integration
- **PyPDF2**: PDF text extraction
- **FPDF2**: PDF generation for exports
- **python-dotenv**: Environment variable management

### Architecture
- **Document Processing**: Handles multiple file formats with robust error handling
- **AI Integration**: Secure API communication with OpenRouter
- **State Management**: Streamlit session state for persistent data across interactions
- **Export System**: Multiple format support with error handling
- **UI/UX**: Responsive design with custom CSS styling

### Error Handling
- Comprehensive validation for API keys and file formats
- Graceful degradation when AI services are unavailable
- User-friendly error messages with troubleshooting guidance
- Automatic retry mechanisms for transient failures

## 📊 Features in Detail

### Summary Types

#### 1. Comprehensive Summary
- Research objectives and questions
- Methodology overview
- Key findings and results
- Conclusions and implications
- Limitations and future work suggestions

#### 2. Key Insights
- Novel contributions to the field
- Significant experimental results
- Practical implications
- Breakthrough findings
- Impact on existing knowledge

#### 3. Methodology Summary
- Experimental design details
- Data collection methods
- Analytical approaches
- Statistical techniques
- Validation procedures

### Comparative Analysis Features
- **Agreement Detection**: Identifies consistent findings across papers
- **Contradiction Analysis**: Highlights conflicting results and interpretations
- **Gap Identification**: Reveals unexplored research areas
- **Methodological Comparison**: Analyzes different approaches used
- **Synthesis Generation**: Creates coherent overview of multiple studies

## 🔒 Security & Privacy

- **API Key Security**: Environment variables for secure key storage
- **Data Privacy**: No data stored permanently, processing happens in memory
- **File Handling**: Temporary files are automatically cleaned up
- **Error Logging**: Minimal logging to protect sensitive information

## 🚨 Troubleshooting

### Common Issues

#### 1. API Key Error (401)
```
❌ Authentication failed: Invalid OpenRouter API key
```
**Solution**: 
- Verify your API key is correct and starts with `sk-or-`
- Check that the key is properly set in the `.env` file
- Ensure you have sufficient credits on your OpenRouter account

#### 2. PDF Processing Error
```
❌ Error reading PDF: [specific error]
```
**Solution**:
- Ensure PDF is not password-protected
- Try converting PDF to text format first
- Check that the PDF contains extractable text (not just images)

#### 3. Rate Limit Error (429)
```
❌ Rate Limit Error: Too many requests
```
**Solution**:
- Wait a few minutes before trying again
- Consider upgrading your OpenRouter plan for higher limits
- Process fewer papers simultaneously

#### 4. Large File Issues
**Solution**:
- Split large papers into smaller sections
- Use text format instead of PDF when possible
- Check file size limits (recommended < 10MB per file)

### Performance Optimization
- **File Size**: Keep individual files under 10MB for best performance
- **Batch Processing**: Process papers one at a time to avoid rate limits
- **Network**: Ensure stable internet connection for API calls

## 🎯 Best Practices

### For Best Results
1. **Quality Input**: Use well-formatted, text-extractable PDFs
2. **Clear Titles**: Ensure paper titles are descriptive
3. **Appropriate Length**: Works best with papers 5-50 pages long
4. **Academic Format**: Optimized for standard academic paper structure

### Usage Tips
- Start with 1-2 papers to test functionality
- Use different summary types for different purposes
- Save important results immediately after generation
- Export data regularly to avoid loss

## 🔄 Updates & Roadmap

### Current Version: 1.0
- Basic summarization and comparison functionality
- OpenRouter integration
- Multi-format file support
- Export capabilities

### Planned Features
- **Advanced Analytics**: Citation analysis, keyword extraction
- **Batch Processing**: Upload and process multiple papers simultaneously
- **Custom Models**: Support for additional AI models
- **Collaboration**: Share results with team members
- **Database Integration**: Permanent storage options
- **API Access**: Programmatic access to summarization features

## 📈 Monitoring & Analytics

### Built-in Statistics
- Papers processed
- Success rates
- Content volume metrics
- Processing time tracking

### Usage Patterns
The application tracks basic usage statistics to help improve performance:
- File format preferences
- Summary type usage
- Error frequency
- Processing times

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Set up environment variables
6. Run: `streamlit run streamlit_app.py`

### Code Structure
- **Main App**: `streamlit_app.py` contains all core functionality
- **Modular Design**: Classes for document processing, AI integration, and UI components
- **Error Handling**: Comprehensive exception handling throughout
- **Documentation**: Inline comments and docstrings

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Authors**: Urja Sahni (102215084) & Sahil Kumar Sahoo (102215179)
- **AI Provider**: OpenRouter for API access
- **Model**: DeepSeek for advanced language processing
- **Framework**: Streamlit for the web interface

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the [OpenRouter documentation](https://openrouter.ai/docs)
3. Open an issue in the project repository
4. Contact the development team

---

**Happy Researching! 🔬📚**
