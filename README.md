# \# Design, Evaluation, and Optimization of a RAG System for Long-Form Technical Documents

# 

# An adaptive Retrieval-Augmented Generation pipeline over a 400+ page ML textbook, with self-correcting retrieval and RAGAS-based evaluation.

# 

# \## What it does

# (one paragraph: ask questions about the book in plain English, get grounded answers with sources; system grades its own retrieval and retries with rewritten queries when it fails)

# 

# \## Architecture

# (pipeline diagram / flow: PDF -> cleaning -> chunking -> embeddings -> FAISS -> graded retrieval -> query rewriting -> grounded generation)

# 

# \## The problem it solves

# (the k-means story: plain RAG fails when question vocabulary differs from corpus vocabulary; show before/after example)

# 

# \## Results

# See \[RESULTS.md](RESULTS.md). Headline: on vocabulary-mismatched queries, context precision 0.39 -> 0.76, recall 0.29 -> 0.67, refusals 5/10 -> 2/10.

# 

# \## How to run

# (setup steps: clone, venv, pip install -r requirements.txt, .env with GOOGLE\_API\_KEY, place corpus PDF in data/, run ingest, run rag)

# 

# \## Limitations \& future work

# (OCR noise in corpus; 2/10 residual hard-query failures; latency cost of grading; future: batched grading, MMR, cross-encoder reranking, embedding model comparison)

