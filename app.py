import os
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# -----------------------------------
# 2. Load embedding model
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# 3. Load FAISS vector database
# -----------------------------------

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# -----------------------------------
# 4. Create retriever
# -----------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------------
# 5. Connect to Hugging Face LLM
# -----------------------------------

client = InferenceClient(
    token=HF_TOKEN
)


# -----------------------------------
# 6. RAG function
# -----------------------------------

def ask_question(question):

    if not question.strip():
        return "Please enter a question."

    # Retrieve relevant chunks
    documents = retriever.invoke(question)

    # Combine retrieved context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create RAG prompt
    prompt = f"""
You are a helpful Evergreen College AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:
"I don't know based on the provided document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate response
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512,
        temperature=0.2
    )

    return response.choices[0].message.content


# -----------------------------------
# 7. Gradio chatbot function
# -----------------------------------

def chatbot(question):
    return ask_question(question)


# -----------------------------------
# 8. Gradio UI
# -----------------------------------

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        label="Your Question",
        placeholder="Ask something about Evergreen College..."
    ),
    outputs=gr.Textbox(
        label="AI Answer"
    ),
    title="Evergreen College AI Assistant",
    description="Ask questions based on the Evergreen College document."
)


# -----------------------------------
# 9. Launch application
# -----------------------------------
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)