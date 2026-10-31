import streamlit as st
import os
import tempfile
import time
import pandas as pd
import json

from dotenv import load_dotenv

# LangChain
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)

# Google Generative AI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# DOCX
from docx import Document as DocxDocument

st.set_page_config(
    page_title="LegalBuddy",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide Streamlit's default header and footer
# Customize Streamlit header and footer
st.markdown("""
    <style>
        /* Keep header visible so sidebar toggle works */
        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Hide footer */
        footer {
            display: none;
        }

        /* Hide Streamlit menu */
        #MainMenu {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)


# Load environment variables
load_dotenv()

# Custom CSS for navbar and general styling
st.markdown("""
<style>
    /* Reset default Streamlit styles */
    .stMarkdown {
        background: transparent !important;
        border: none !important;
    }
    
    .stMarkdown > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Navbar style */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: black;
        padding: 0.7rem 2rem;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .navbar-content {
        display: flex;
        align-items: center;
    }
    
    .navbar-brand {
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
        padding: 0;
        line-height: 1.5;
    }
    
    /* Main content area */
    .main-content {
        margin-top: 80px;
        padding: 20px;
    }
    
    /* Remove default white backgrounds */
    .stFileUploader > div {
        background-color: transparent !important;
        border: none !important;
    }
    
    .stFileUploader > div > div {
        background-color: transparent !important;
        border: 1px solid #0f1629 !important;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        height: 120px;
        background-color: #0f1629;
        border: none;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        background-color: #1c2a4e;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: transparent !important;
        border: none !important;
    }
    
    .streamlit-expanderContent {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Remove white boxes from chat history */
    .element-container {
        background-color: transparent !important;
    }
    
    /* Additional styling to remove any remaining white boxes */
    div[data-testid="stMarkdownContainer"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
    <div class="navbar-content">
        <h1 class="navbar-brand">📚 LegalBuddy</h1>
    </div>
</div>
<div class="main-content">
""", unsafe_allow_html=True)

# Initialize session state variables
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# --------------------------- ENHANCED SYSTEM PROMPT ---------------------------
LEGAL_EXTRACTION_PROMPT = """You are an expert legal document analyst. Your task is to extract and categorize key details from the given legal document, ensuring accuracy and completeness. 
Even if the exact term is not mentioned, identify similar phrases or concepts that convey the same meaning. If no relevant information is found, explicitly state "N/A".

### 🔍 Extraction Guidelines:

#### 1️⃣ Entities & Contact Details
   - Identify all parties involved (individuals, companies, organizations).
   - Extract full legal names.
   - Capture addresses, emails, and phone numbers.

#### 2️⃣ Contract Start Date & End Date
   - Locate the contract’s effective date (start date).
   - Identify the expiration or termination date.
   - Note any key milestone dates (e.g., renewal deadlines, review periods).

#### 3️⃣ Scope of Agreement
   - Clearly define the document’s purpose.
   - Highlight key obligations, deliverables, and services mentioned.
   - Extract any relevant exclusions or limitations.

#### 4️⃣ Service Level Agreement (SLA)
   - Extract performance metrics, response times, and service standards.
   - Identify any penalties for SLA breaches.

#### 5️⃣ Penalty Clauses
   - Identify conditions that trigger penalties.
   - Extract monetary/legal consequences for non-compliance.
   - Define what constitutes a breach or violation.

#### 6️⃣ Confidentiality Clause
   - Identify confidentiality obligations and restrictions.
   - Extract the duration and scope of confidentiality terms.

#### 7️⃣ Renewal & Termination Clause
   - Extract conditions for renewal (auto-renewal, renegotiation terms).
   - Identify termination clauses (grounds for termination).
   - Note any required notice periods.

#### 8️⃣ Commercials / Payment Terms
   - Extract payment terms, pricing structures, and invoicing details.
   - Identify due dates, penalties for late payments, and refund policies.

#### 9️⃣ Risks & Assumptions
   - Identify potential risks associated with the agreement.
   - Extract any stated mitigation strategies or underlying assumptions.

If any section is missing, explicitly return "N/A".

---

### 📜 Document Context:
<context>
{context}
</context>

### 🔍 Extraction Task:
Extract and categorize all legal information following the above structure. If specific terms are not found, look for synonyms or related phrases. If no relevant information exists, return "N/A".
"""

# Prompts for different extraction tasks
extraction_prompt = ChatPromptTemplate.from_template(
    f"""
{LEGAL_EXTRACTION_PROMPT}

📜 *Document Context*:
<context>
{{context}}
</context>

🔍 *Extraction Task*: Extract and categorize all available legal information from the document.
"""
)

qa_prompt = ChatPromptTemplate.from_template(
    f"""
You are a legal document assistant. Provide precise and contextual answers.

📜 *Document Context*:
<context>
{{context}}
</context>

🔍 *User Question*: {{input}}

Provide a clear, concise answer based strictly on the document context.
"""
)

# --------------------------- DOCUMENT PROCESSOR ---------------------------
def process_uploaded_file(uploaded_file):
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()

    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Process different file types
    try:
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
        elif file_extension == 'txt':
            loader = TextLoader(tmp_path)
            documents = loader.load()
        elif file_extension == 'docx':
            documents = []
            doc = DocxDocument(tmp_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    documents.append(Document(page_content=para.text))
        elif file_extension == 'csv':
            loader = CSVLoader(tmp_path)
            documents = loader.load()
        elif file_extension == 'xlsx':
            loader = UnstructuredExcelLoader(tmp_path)
            documents = loader.load()
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return []
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

    return documents

def vector_embedding(documents):
    with st.spinner("Processing documents..."):
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.success("Documents processed successfully!")

def convert_to_json(extraction_text):
    """Convert the extracted text into a structured JSON format"""
    sections = {
        "entities_and_contacts": {},
        "contract_timeline": {},
        "scope": "",
        "sla_clauses": [],
        "penalty_clauses": [],
        "confidentiality": {},
        "renewal_termination": {},
        "commercial_terms": {},
        "risks_assumptions": []
    }
    
    # Parse the extraction text and populate sections
    current_section = None
    for line in extraction_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if any(key.replace('_', ' ').upper() in line.upper() for key in sections.keys()):
            current_section = next(key for key in sections.keys() if key.replace('_', ' ').upper() in line.upper())
            continue
            
        if current_section and line:
            if isinstance(sections[current_section], dict):
                # Split on first colon for key-value pairs
                if ':' in line:
                    key, value = line.split(':', 1)
                    sections[current_section][key.strip()] = value.strip()
            elif isinstance(sections[current_section], list):
                sections[current_section].append(line)
            else:
                sections[current_section] = line
    
    return sections

# --------------------------- INITIALIZE LLM ---------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

# --------------------------- MAIN APP LAYOUT ---------------------------
st.markdown("## 👋 Hello! What do you want to know today?")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload your legal documents (PDF, DOCX, TXT, CSV, XLSX, etc.)", 
    type=['pdf', 'txt', 'docx', 'csv', 'xlsx'], 
    accept_multiple_files=True
)

if uploaded_files:
    new_files = [f.name for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        st.write("New files to process:", new_files)
        if st.button("🚀 Process Documents", use_container_width=True):
            all_documents = []
            for uploaded_file in uploaded_files:
                documents = process_uploaded_file(uploaded_file)
                all_documents.extend(documents)
            if all_documents:
                vector_embedding(all_documents)
                st.session_state.processed_files = [f.name for f in uploaded_files]

# Functionality Options in Sidebar
if st.session_state.vectors is not None:
    with st.sidebar:
        st.markdown("## 📂 Document Actions")
        
        # Card 1 - Document Extraction
        if st.button("🔍 Extract Key Details\n\nAutomatic extraction of critical legal information"):
            st.session_state.selected_option = 'extraction'

        # Card 2 - Chat with Documents
        if st.button("💬 Chat with Docs\n\nAsk specific questions about your documents"):
            st.session_state.selected_option = 'qa'

        # Card 3 - Maintain Chat History
        if st.button("📜 Chat History\n\nReview previous conversations"):
            st.session_state.selected_option = 'history'

        # Card 4 - Document Summary
        if st.button("📄 Generate Summary\n\nQuick overview of document contents"):
            st.session_state.selected_option = 'summary'

    # --------------------------- MAIN FUNCTIONALITY ---------------------------
    if st.session_state.selected_option == 'extraction':
        if "extraction_result" not in st.session_state:
            with st.spinner("Extracting key details..."):
                document_chain = create_stuff_documents_chain(llm, extraction_prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start = time.process_time()
                response = retrieval_chain.invoke({'input': 'Extract all key legal information from the document'})
                elapsed = time.process_time() - start

                # Convert to JSON
                json_data = convert_to_json(response['answer'])

                # Store the result in session state
                st.session_state.extraction_result = {
                    "answer": response['answer'],
                    "json_data": json_data,
                    "elapsed": elapsed,
                    "context": response.get("context", [])
                }

        # Retrieve from session state
        extraction_result = st.session_state.extraction_result

        st.success(f"Analysis completed in {extraction_result['elapsed']:.2f} seconds!")
        st.subheader("📑 Legal Document Insights")
        st.write(extraction_result['answer'])

        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(extraction_result['json_data'], indent=2),
                file_name="legal_extraction.json",
                mime="application/json"
            )
        with col2:
            st.download_button(
                label="📥 Download Text",
                data=extraction_result['answer'],
                file_name="legal_extraction.txt",
                mime="text/plain"
            )

        with st.expander("📚 Source Document Sections"):
            for i, doc in enumerate(extraction_result["context"]):
                st.markdown(f"Section {i+1}")
                st.write(doc.page_content)
                st.divider()


    elif st.session_state.selected_option == 'qa':
        prompt1 = st.text_input("🔍 Ask a specific question about your documents")
        if prompt1:
            # Store user question in chat history
            st.session_state.chat_history.append(f"User: {prompt1}")
            
            with st.spinner("Finding answer..."):
                document_chain = create_stuff_documents_chain(llm, qa_prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                elapsed = time.process_time() - start
                
                # Store AI response in chat history
                st.session_state.chat_history.append(f"AI: {response['answer']}")
                
                st.success(f"Answer found in {elapsed:.2f} seconds!")
                st.subheader("💡 Answer")
                st.write(response['answer'])
                
                with st.expander("📚 Relevant Document Sections"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"Section {i+1}")
                        st.write(doc.page_content)
                        st.divider()

    elif st.session_state.selected_option == 'history':
        st.subheader("📜 Chat History")
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                st.write(message)
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.experimental_rerun()
        else:
            st.write("No chat history available.")
    
    elif st.session_state.selected_option == 'summary':
        if "summary" in st.session_state:  # Check if summary exists
            st.subheader("📄 Document Summary")
            st.write(st.session_state["summary"])  # Display stored summary
        else:
            with st.spinner("Generating document summary..."):
                summary_prompt = ChatPromptTemplate.from_template("""
                You are an expert document summarizer. Provide a comprehensive yet concise summary of the document.

                Key Summary Requirements:
                - Capture the main purpose and context of the document
                - Highlight key points and critical information
                - Maintain objectivity and clarity
                - Be precise and avoid unnecessary details

                📜 Document Context:
                <context>
                {context}
                </context>

                Generate a clear, structured summary that captures the essence of the document.
                """)

                document_chain = create_stuff_documents_chain(llm, summary_prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': 'Generate a comprehensive summary of the entire document'})
                elapsed = time.process_time() - start

                st.session_state["summary"] = response['answer']  # Store summary

                st.success(f"Summary generated in {elapsed:.2f} seconds!")
                st.subheader("📄 Document Summary")
                st.write(st.session_state["summary"])  # Display stored summary

        # Optional: Add download button for summary
        st.download_button(
            label="📥 Download Summary",
            data=st.session_state["summary"],
            file_name="document_summary.txt",
            mime="text/plain"
        )


# --------------------------- FOOTER ---------------------------
st.markdown("---")
st.markdown("Made with ❤ by LegalBuddy Team")