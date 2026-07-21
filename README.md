# Design, Evaluation, and Optimization of a RAG System for Long-Form Technical Documents

🔗 **[Try the live demo](https://adaptive-rag-pipeline-deploy-j9dwp8onxhzy958tuno8u2.streamlit.app/)**

An adaptive Retrieval-Augmented Generation pipeline over a 400+ page ML textbook, with self-correcting retrieval and RAGAS-based evaluation.

## What it does

Ask questions about Tom Mitchell's *Machine Learning* textbook in plain English and get grounded answers with source passages. The system grades its own retrieval quality on every query: if the retrieved chunks are not relevant enough, it rewrites the question into textbook vocabulary and retries before answering. Answers are generated strictly from retrieved content, and the system says "I don't know" rather than hallucinate.

## Architecture

PDF → text extraction (pypdf) → cleaning (front-matter removal) → chunking (1000 chars, 200 overlap) → embeddings (all-MiniLM-L6-v2) → FAISS vector store → graded retrieval (LLM relevance judge) → query rewriting on low relevance → grounded generation (Gemini Flash-Lite) → RAGAS evaluation

## The problem it solves

Plain RAG fails when a question's vocabulary differs from the corpus vocabulary. Example from this project: asking "What is the k-means algorithm?" retrieved cover-page junk and k-nearest-neighbor chunks, because the book discusses the concept as "estimating the means of k Gaussians" and never uses the word k-means. The adaptive layer detects this failure (relevance grading), bridges the vocabulary gap (query rewriting), and passes both phrasings to generation.

## Results

See [RESULTS.md](RESULTS.md) for full tables.

Headline: on vocabulary-mismatched queries, the adaptive layer improved context precision from 0.39 to 0.76, context recall from 0.29 to 0.67, and cut refusals from 5/10 to 2/10, versus baseline RAG.

## How to run

1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Create a `.env` file with `GOOGLE_API_KEY=your-key`
4. Place your corpus PDF in `data/`
5. `python src/ingest.py` to build the FAISS index
6. `python src/adaptive_rag.py` to ask questions, or `streamlit run app.py` for the web UI

## Limitations and future work

- Corpus is an OCR'd scan: mathematical notation is garbled, so evaluation targets conceptual prose
- 2/10 hard queries still fail: one unrecoverable retrieval miss, one over-conservative generation
- Grading adds 4-8s latency per query; future: batched grading to cut LLM calls
- Future experiments: MMR retrieval, cross-encoder reranking, embedding model comparison