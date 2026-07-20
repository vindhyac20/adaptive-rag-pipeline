# Results



## Setup

- Corpus: Machine Learning (Tom Mitchell), 409 content pages, 1,348 chunks (1000 chars, 200 overlap)

- Embeddings: all-MiniLM-L6-v2 (384-dim), FAISS vector store

- Generation: Gemini 3.1 Flash-Lite

- Evaluation: RAGAS (faithfulness, answer relevancy, context precision, context recall)

- Test sets: 20 standard questions (textbook phrasing), 10 hard questions (casual/mismatched vocabulary)



## Standard test set (20 questions)



| Metric | Baseline RAG | Adaptive RAG |

|---|---|---|

| Faithfulness | 0.941 | 0.944 |

| Answer relevancy | 0.907 | 0.863 |

| Context precision | 0.667 | 0.690 |

| Context recall | 0.850 | 0.825 |

| Avg latency | ~1.0s | ~4.9s |

| Rewrites triggered | â€” | 0/20 |



On well-phrased questions, both systems perform comparably: first-pass retrieval succeeds, so the adaptive layer adds cost without benefit.



## Hard test set (10 vocabulary-mismatched questions)



| Metric | Baseline RAG | Adaptive RAG (final) |

|---|---|---|

| Faithfulness | 0.729 | 0.771 |

| Answer relevancy | 0.206 | 0.321 |

| Context precision | 0.385 | **0.759** |

| Context recall | 0.286 | **0.667** |

| Refusals | 5/10 | 2/10 |

| Rewrites triggered | â€” | 7/10 |



Baseline RAG collapses when question vocabulary differs from the corpus. The adaptive layer (relevance grading + query rewriting + generation-side vocabulary bridging) nearly doubles context precision and recall, and cuts refusals from 5 to 2.



## Key findings

1. Retrieval quality, not generation, was the bottleneck (baseline faithfulness stayed high while precision collapsed).

2. Fixing retrieval alone was insufficient: recovered chunks used textbook vocabulary the question lacked, so strictly grounded generation still refused. Appending the rewritten query at generation time resolved this.

3. Residual failures (2/10): one unrecoverable retrieval miss, one over-conservative generation despite correct context.

4. Cost: adaptive adds ~4-8s latency per query from grading calls (mitigation: batched grading â€” future work).


