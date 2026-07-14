import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load the saved index (no re-embedding of the book!)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

# 2. The LLM
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# 3. The prompt template - the anti-hallucination contract
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

# 4. The RAG function
def ask(question):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content

# 5. Test it
if __name__ == "__main__":
    q = "Who won the last cricket world cup?"
    print("Q:", q)
    print("A:", ask(q))