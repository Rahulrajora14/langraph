from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.tools import tool

# -----------------------------
# Load PDF
# -----------------------------

loader = PyPDFLoader("data/ml_book.pdf")

documents = loader.load()

# -----------------------------
# Split
# -----------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200

)

chunks = splitter.split_documents(documents)

# -----------------------------
# Embedding
# -----------------------------

embedding_model = GoogleGenerativeAIEmbeddings(

    model="models/embedding-001"

)

vector_store = FAISS.from_documents(

    chunks,

    embedding_model

)

retriever = vector_store.as_retriever(

    search_kwargs={"k":4}

)

# -----------------------------
# Tool
# -----------------------------

@tool
def rag_tool(query:str)->str:
    """
    Search the uploaded PDF and answer user questions.
    """

    docs = retriever.invoke(query)

    context = "\n\n".join(

        doc.page_content

        for doc in docs

    )

    return context