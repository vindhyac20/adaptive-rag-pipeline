import json
import time
import sys
sys.path.append("src")

from adaptive_rag import adaptive_ask

with open("evaluation/hard_questions.json", encoding="utf-8") as f:
    test_set = json.load(f)

print(f"Running ADAPTIVE evaluation on {len(test_set)} HARD questions...\n")

results = []
for i, item in enumerate(test_set, 1):
    q = item["question"]
    print(f"[{i}/{len(test_set)}] {q}")

    start = time.time()
    result = adaptive_ask(q)
    elapsed = time.time() - start

    results.append({
        "question": q,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": item["ground_truth"],
        "latency_seconds": round(elapsed, 2),
        "rewritten_query": result["rewritten_query"],
    })
    flag = " [REWRITE]" if result["rewritten_query"] else ""
    print(f"    done in {elapsed:.1f}s{flag}\n")

    time.sleep(30)

with open("evaluation/eval_results_hard_adaptive.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Saved to evaluation/eval_results_hard_adaptive.json")