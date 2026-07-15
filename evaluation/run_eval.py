import json
import time
import sys
sys.path.append("src")

from rag import ask, vectorstore

# Load the test questions
with open("evaluation/test_questions.json", encoding="utf-8") as f:
    test_set = json.load(f)

print(f"Running evaluation on {len(test_set)} questions...\n")

results = []
for i, item in enumerate(test_set, 1):
    q = item["question"]
    print(f"[{i}/{len(test_set)}] {q}")

    start = time.time()
    docs = vectorstore.similarity_search(q, k=4)
    contexts = [d.page_content for d in docs]
    result = ask(q)
    elapsed = time.time() - start

    results.append({
        "question": q,
        "answer": result["answer"],
        "contexts": contexts,
        "ground_truth": item["ground_truth"],
        "latency_seconds": round(elapsed, 2),
    })
    print(f"    done in {elapsed:.1f}s\n")

    time.sleep(5)  # pacing: stay well under 15 requests/minute

with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved to evaluation/eval_results.json")