import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import tempfile
import time

load_dotenv()

# Custom CSS for the cards
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        height: 200px;
        background-color: #f0f2f6;
        border: none;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #e0e2e6;
    }
    div.stButton > button p {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #0f1629;
    }
    div.stButton > button small {
        font-size: 16px;
        color: #485164;
    }
    .selected {
        border: 2px solid #ff4b4b !important;
        background-color: #ffeaea !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 LegalBuddy")
st.markdown("### Your AI-powered Legal Document Assistant")

# Initialize session state
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

## load the GROQ And OpenAI API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Prompts
qa_prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

summary_prompt = ChatPromptTemplate.from_template(
"""
Please provide a comprehensive summary of the following document context.
Focus on the key points and main ideas.
<context>
{context}
</context>
"""
)

def process_uploaded_file(uploaded_file):
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Process the PDF file
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Clean up the temporary file
    os.unlink(tmp_path)
    
    return documents

def vector_embedding(documents):
    with st.spinner("Processing documents..."):
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.success("Documents processed successfully!")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# File uploader
uploaded_files = st.file_uploader("Upload your PDF documents", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    new_files = [f.name for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        st.write("New files to process:", new_files)
        if st.button("🚀 Process Documents", use_container_width=True):
            all_documents = []
            for uploaded_file in uploaded_files:
                documents = process_uploaded_file(uploaded_file)
                all_documents.extend(documents)
            vector_embedding(all_documents)
            st.session_state.processed_files = [f.name for f in uploaded_files]

# Show options and functionality after processing
if st.session_state.vectors is not None:
    # Create two columns for the cards
    col1, col2 = st.columns(2)

    # Card 1 - Document Summary
    with col1:
        if st.button("📋 Document Summary\n\nGet a comprehensive summary of your legal documents"):
            st.session_state.selected_option = 'summary'

    # Card 2 - Ask Questions
    with col2:
        if st.button("❓ Ask Questions\n\nGet specific answers from your documents"):
            st.session_state.selected_option = 'qa'

    # Main functionality based on selection
    if st.session_state.selected_option == 'summary':
        with st.spinner("Generating summary..."):
            document_chain = create_stuff_documents_chain(llm, summary_prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            response = retrieval_chain.invoke({'input': 'Summarize the document'})
            
            st.success(f"Summary generated in {time.process_time() - start:.2f} seconds!")
            
            st.subheader("📑 Document Summary")
            st.write(response['answer'])
            
            with st.expander("📚 View Source Sections"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"*Section {i+1}*")
                    st.write(doc.page_content)
                    st.divider()

    elif st.session_state.selected_option == 'qa':
        prompt1 = st.text_input("🔍 Enter your question about the documents")
        
        if prompt1:
            with st.spinner("Finding answer..."):
                document_chain = create_stuff_documents_chain(llm, qa_prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                
                st.success(f"Answer found in {time.process_time() - start:.2f} seconds!")
                
                st.subheader("💡 Answer")
                st.write(response['answer'])
                
                with st.expander("📚 View Relevant Sections"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"*Section {i+1}*")
                        st.write(doc.page_content)
                        st.divider()

# Add a footer
st.markdown("---")
st.markdown("Made with ❤ by LegalBuddy Team")