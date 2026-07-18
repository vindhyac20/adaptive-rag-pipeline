import json
import time
import sys
sys.path.append("src")

from rag import ask, vectorstore

with open("evaluation/hard_questions.json", encoding="utf-8") as f:
    test_set = json.load(f)

print(f"Running BASELINE evaluation on {len(test_set)} HARD questions...\n")

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

    time.sleep(5)

with open("evaluation/eval_results_hard_baseline.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved to evaluation/eval_results_hard_baseline.json")