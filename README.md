# 📚 Research Papers RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain that allows you to ask questions about your research papers and get AI-generated answers based on their content.

## ✨ Features

- **PDF Processing**: Automatically processes academic papers and research documents in PDF format
- **Embeddings Generation**: Creates vector embeddings of document content using HuggingFace's MiniLM model
- **Semantic Search**: Retrieves the most relevant document chunks based on your query
- **AI-Powered Answers**: Uses Llama3 via Groq API to generate accurate, contextual responses
- **Source References**: Shows the exact document chunks used to generate each answer
- **Modern UI**: Clean, responsive interface with dark mode support

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Groq API key
- HuggingFace token
- Research papers in PDF format

### Installation

Create a `.env` file in the project root with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Create a directory for your research papers:
```bash
mkdir research_papers
```

Add your PDF research papers to the `research_papers` directory.

### Running the Application

```bash
streamlit run app.py
```

Navigate to the URL provided by Streamlit (typically http://localhost:8501) to interact with the application.

## 📋 Usage Guide

1. **Process Documents**: 
   - Click the "Process Documents" button to analyze your PDFs and create embeddings
   - Wait for the processing to complete (this may take some time depending on the number and size of your documents)

2. **Ask Questions**:
   - Enter your question about the research papers in the text area
   - Click "Submit Question" to generate an answer
   - Review the answer and the source documents that were used to generate it

3. **Analyze Sources**:
   - Expand the "View Source Documents" section to see exactly which parts of your papers were used to create the answer
   - Use this to verify the accuracy and context of the generated response

## 🔧 Customization

### Adjusting Document Processing

To modify how documents are processed, you can adjust these parameters in the code:

```python
# Change chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust this value
    chunk_overlap=200  # Adjust this value
)

# Change the number of relevant chunks retrieved
retriever = vectors.as_retriever(
    search_kwargs={"k": 5}  # Adjust this value
)
```

### Using Different Models

You can modify the LLM or embeddings model by changing:

```python
# Change the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use a different model

# Change the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")  # Use a different model
```

## 🔒 Privacy and Data Handling

- All document processing happens locally on your machine
- Only your specific queries are sent to the Groq API
- Document embeddings are stored in memory during the session
- No data is permanently stored or shared with third parties

## 🚧 Limitations

- Currently supports PDF files only
- Processing large numbers of documents may be memory-intensive
- The application processes a maximum of 50 documents per session to ensure stability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- Support for additional document formats (DOCX, TXT, etc.)
- Persistent vector storage for faster startup
- Custom prompt templates for different types of questions
- Document metadata filtering
- Citation generation for answers
- Multiple LLM provider options

---
## Output



https://github.com/user-attachments/assets/59ac60ce-4036-4276-96b0-23028f9d0693




Built with ❤️ using Streamlit, LangChain, FAISS, and Groq
