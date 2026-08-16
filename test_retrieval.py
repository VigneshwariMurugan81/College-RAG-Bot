from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore"

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Ask a question
question = "What is Evergreen College?"

# Retrieve relevant chunks
results = retriever.invoke(question)

print("\nRelevant information:\n")

for i, document in enumerate(results):
    print(f"--- Result {i + 1} ---")
    print(document.page_content)
    print()