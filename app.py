import streamlit as st
import sys
sys.path.append("src")

from adaptive_rag import adaptive_ask

st.set_page_config(page_title="Ask Mitchell's ML Textbook", page_icon="📘")

st.title("📘 Ask Mitchell's Machine Learning Textbook")
st.caption("Built on Tom Mitchell's Machine Learning textbook, with adaptive retrieval and RAGAS evaluation")
st.write("Ask a question in your own words. The system finds relevant passages, checks if they're actually useful, and rewrites the question if the first search fails.")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        result = adaptive_ask(question)

    if result["rewritten_query"]:
        st.info(f"🔄 First search wasn't good enough. Rewrote the question as: *{result['rewritten_query']}*")

    st.subheader("Answer")
    st.markdown(result["answer"])

    if result["sources"]:
        st.subheader("Sources used")
        for i, src in enumerate(result["sources"], 1):
            with st.expander(f"Source {i}"):
                st.markdown(src)