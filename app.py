import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import shutil
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import gc

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Store API key securely in .env

# Flag to switch between Local LLM and Groq API
USE_LOCAL_LLM = False  # Set to True when switching to a local model in the future


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split extracted text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    """Store text chunks in ChromaDB for efficient retrieval."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectorstore = Chroma.from_texts(
        texts=text_chunks, embedding=embeddings, persist_directory="chroma_db"
    )
    vectorstore.persist()
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create a conversation chain with Groq API or local LLM."""
    if USE_LOCAL_LLM:
        llm = Ollama(model="deepseek-r1:1.5b")  # Use local LLM
    else:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY,
                    model_name="llama3-70b-8192")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
def reset_vectorstore():
    # Reset the ChromaDB by deleting the old persisted directory
    if os.path.exists("chroma_db"):
        # Close any open resources
        try:
            # Make sure Chroma is properly disposed of
            gc.collect()  # Force garbage collection to free up resources
            shutil.rmtree("chroma_db")  # Remove the old vector store
        except PermissionError as e:
            print(f"Error while deleting chroma_db: {e}")
    # Create a new vector store each time
    os.makedirs("chroma_db", exist_ok=True)

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF Files", page_icon="📚")
    st.header("Chat with Your PDFs 📚")
    
    query = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                # Reset ChromaDB to ensure fresh storage for every upload
                reset_vectorstore()
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore  # Store in session state
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Documents processed successfully! Ready for Q&A.")

    # Handle user queries
    if query:
        if "conversation" not in st.session_state:
            st.error("Please upload and process documents first.")
        else:
            with st.spinner("Generating response..."):
                response = st.session_state.conversation.run(query)
                st.write("### Answer:")
                st.write(response)


if __name__ == "__main__":
    main()
