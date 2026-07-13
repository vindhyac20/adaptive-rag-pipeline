from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load the book
reader = PdfReader("data/MachineLearningTomMitchell.pdf")
full_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        full_text += page_text + "\n"
print("Total characters:", len(full_text))

# Step 2: Chunk it
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.create_documents(
    [full_text],
    metadatas=[{"source": "MachineLearningTomMitchell.pdf"}],
)
print("Chunks:", len(docs))

# Step 3: Embed all chunks and build the FAISS index
print("Embedding chunks... (this takes a few minutes on CPU)")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 4: Save the index to disk
vectorstore.save_local("faiss_index")
print("Index saved to faiss_index/")

# Step 5: Test it with a real question
results = vectorstore.similarity_search("How does the FIND-S algorithm work?", k=3)
print("\n---- Top retrieved chunk ----")
print(results[0].page_content[:400])