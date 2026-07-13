from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

v1 = model.embed_query("The cat sat on the mat")
v2 = model.embed_query("A kitten was resting on the rug")
v3 = model.embed_query("Stock prices fell sharply on Monday")

print("Vector length:", len(v1))
print("First 5 numbers of v1:", v1[:5])

def similarity(a, b):
    return sum(x * y for x, y in zip(a, b))

print("cat-sentence vs kitten-sentence:", similarity(v1, v2))
print("cat-sentence vs stock-sentence:", similarity(v1, v3))