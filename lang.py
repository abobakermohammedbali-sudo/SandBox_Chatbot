import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "AQ.Ab8RN6LRn3MNUnd8ARux4WVJYa0YP57hh_RDzaXuMYgze-YnHQe"

# 1. Ingestion: Create a dummy text file to act as your sandbox knowledge base
with open("knowledge_base.txt", "w") as f:
    f.write("The RAG sandbox is a great environment to test LangChain pipelines. LangChain was founded by Harrison Chase.")

loader = TextLoader("knowledge_base.txt")
docs = loader.load()

# 2. Chunking: Split the document into small pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 3. Embedding & Indexing: Store embeddings in ChromaDB
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# 4. Generation: Set up the Chat Model and RAG Prompt
llm = ChatOpenAI(model="gpt-4o")
system_prompt = (
    "You are a helpful assistant. Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Execution: Combine into a LangChain retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Test the Sandbox
response = rag_chain.invoke({"input": "Who founded LangChain?"})
print("Answer:", response["answer"])
