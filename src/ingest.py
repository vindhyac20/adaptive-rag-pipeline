from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load all pages into one big string
reader = PdfReader("data/MachineLearningTomMitchell.pdf")
full_text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        full_text += page_text + "\n"

print("Total characters in book:", len(full_text))

# Step 2: Split into overlapping chunks (as Documents with metadata)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = splitter.create_documents(
    [full_text],
    metadatas=[{"source": "MachineLearningTomMitchell.pdf"}],
)

print("Number of chunks:", len(docs))
print("---- chunk 100 ----")
print(docs[100].page_content[:300])
print("Metadata:", docs[100].metadata)