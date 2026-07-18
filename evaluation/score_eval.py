import json
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

answer_relevancy.strictness = 1

with open("evaluation/eval_results_adaptive.json", encoding="utf-8") as f:
    results = json.load(f)

dataset = Dataset.from_list(results)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", max_retries=2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings,
    run_config=RunConfig(max_workers=1, max_retries=3, timeout=120),
)

print("\n===== ADAPTIVE SCORES =====")
print(scores)

df = scores.to_pandas()
df.to_csv("evaluation/adaptive_scores.csv", index=False)
print("Per-question scores saved to evaluation/adaptive_scores.csv")