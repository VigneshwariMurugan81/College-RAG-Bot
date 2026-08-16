import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOC_PATH = "document/evergreen_college_profile.docx"
VECTORSTORE_PATH = "vectorstore"

# Load document
text = docx2txt.process(DOC_PATH)

print("Document loaded successfully!")

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.create_documents([text])

print("Number of chunks:", len(chunks))

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# Create FAISS database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS database created!")

# Save database
vectorstore.save_local(VECTORSTORE_PATH)

print("FAISS database saved successfully!")