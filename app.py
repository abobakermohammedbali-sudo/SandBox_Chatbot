import os
import streamlit as st
# Update to use langchain_community
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- STEP 1: STREAMLIT UI LAYOUT SETUP ---
st.set_page_config(page_title="Free Local RAG Sandbox", layout="wide")
st.title("Abobaker RAG Sandbox (Ollama + LangChain)")

# Sidebar for data source uploading
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF reference document", type=["pdf"])
    
    process_button = st.button("Process & Index Document")
    st.caption("Running 100% locally via Ollama. No API quota needed!")

# Initialize session states so data structures persist across UI re-renders
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- STEP 2: PDF DATA PROCESSING & CHROMA VECTOR STORAGE ---
if process_button and uploaded_file:
    with st.spinner("Parsing and indexing PDF document chunks locally..."):
        try:
            # Save uploaded bytes to a temporary file locally to parse with PyPDF
            temp_file_path = f"./temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Parse text out of the raw PDF file
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            # Split continuous text into 1,000-character blocks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)
            
            # Dedicated embedding model designed specifically for RAG
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            st.session_state.vectorstore = Chroma.from_documents(
                documents=text_chunks, 
                embedding=embeddings
            )
            
            # Remove the temporary disk file securely
            os.remove(temp_file_path)
            st.success("Local vector database built successfully! RAG routing active.")
            
        except Exception as e:
            st.error(f"Failed to process document: {str(e)}")


# --- STEP 3: RAG LOGIC & LANGCHAIN EXPRESSION LANGUAGE (LCEL) CHAIN ---
def build_execution_chain():
    # Instantiate the local llama3.2 engine running on your machine for text generation
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    
    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Condition: If a database index exists, assemble the full retrieval RAG chain
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "Answer the user question using ONLY the provided context below. "
                "If you do not know the answer based on the context, say that you do not know.\n\n"
                "Context:\n{context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # FIXED: Using retriever.invoke instead of the outdated retriever.get_relevant_documents
        chain = (
            RunnablePassthrough.assign(context=lambda inputs: combine_docs(retriever.invoke(inputs["question"])))
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        # Fallback condition: If no document is uploaded, route queries to standard LLM chat
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI conversational assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        
    return chain


# --- STEP 4: CONVERSATIONAL UI MANAGEMENT & STREAMING ENGINE ---
# Render historical messages on the screen during page refreshes
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Intercept and process live incoming text query questions
if user_query := st.chat_input("Ask something about your data or type a generic question..."):
    
    # Promptly render the user query block on the interface
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Open assistant message container to host real-time token streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        active_chain = build_execution_chain()
        
        # Trigger stream loop to render words onto the dashboard token-by-token
        full_response = ""
        for chunk in active_chain.stream({
            "question": user_query, 
            "chat_history": st.session_state.chat_history
        }):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
            
        # Clean up by printing final response content cleanly without trailing cursor block
        response_placeholder.markdown(full_response)
        
    # Append transaction pairs to persistent state memory
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=full_response))
