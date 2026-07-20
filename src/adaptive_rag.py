import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- Setup: vector store and LLM ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", max_retries=6)

# --- Grader: judges whether a retrieved chunk is relevant ---
grade_prompt = ChatPromptTemplate.from_template("""
You are grading whether a retrieved document chunk is relevant to a question.
Answer with exactly one word: yes or no.

Question: {question}

Chunk:
{chunk}

Is this chunk relevant to answering the question? Answer yes or no:""")

def grade_chunk(question, chunk):
    messages = grade_prompt.format_messages(question=question, chunk=chunk)
    response = llm.invoke(messages)
    return response.content.strip().lower().startswith("yes")

# --- Rewriter: rephrases a question that retrieved badly ---
rewrite_prompt = ChatPromptTemplate.from_template("""
A search over a machine learning textbook failed to find relevant passages for this question.
Rewrite the question using different technical vocabulary that a textbook might use,
keeping the same meaning. Output ONLY the rewritten question, nothing else.

Original question: {question}

Rewritten question:""")

def rewrite_query(question):
    messages = rewrite_prompt.format_messages(question=question)
    response = llm.invoke(messages)
    return response.content.strip()

# --- Adaptive retrieval: retrieve -> grade -> rewrite & retry once if needed ---
def adaptive_ask_retrieve(question, k=4, min_relevant=3):
    """Retrieve with grading; rewrite and retry once if grading fails."""
    docs = vectorstore.similarity_search(question, k=k)
    relevant = [d for d in docs if grade_chunk(question, d.page_content)]
    rewritten = None

    if len(relevant) < min_relevant:
        rewritten = rewrite_query(question)
        print(f"    [adaptive] low relevance ({len(relevant)}/{k}), retrying with: {rewritten}")
        docs2 = vectorstore.similarity_search(rewritten, k=k)
        relevant2 = [d for d in docs2 if grade_chunk(question, d.page_content)]
        if len(relevant2) > len(relevant):
            relevant = relevant2

    return relevant, rewritten

# --- Answer prompt: same contract as baseline rag.py ---
answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

def adaptive_ask(question, k=4, min_relevant=3):
    """Full adaptive RAG: graded retrieval with retry, then grounded generation.
    When a rewrite fired, the rewritten phrasing is appended to the question at
    generation time so the model can bridge vocabulary between question and context."""
    relevant, rewritten = adaptive_ask_retrieve(question, k=k, min_relevant=min_relevant)

    if not relevant:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "contexts": [],
            "rewritten_query": rewritten,
        }

    context = "\n\n".join(doc.page_content for doc in relevant)
    generation_question = question
    if rewritten:
        generation_question = f"{question}\n(Equivalently phrased: {rewritten})"
    messages = answer_prompt.format_messages(context=context, question=generation_question)
    response = llm.invoke(messages)
    return {
        "answer": response.content,
        "sources": [doc.page_content[:200] for doc in relevant],
        "contexts": [doc.page_content for doc in relevant],
        "rewritten_query": rewritten,
    }

# --- Smoke test ---
if __name__ == "__main__":
    q = "Why does a spam filter multiply probabilities of individual words together?"
    result = adaptive_ask(q)
    print("Q:", q)
    if result["rewritten_query"]:
        print("(rewritten as:", result["rewritten_query"] + ")")
    print("A:", result["answer"])