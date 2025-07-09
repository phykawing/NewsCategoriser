import streamlit as st
import pandas as pd
from news_pipeline import generate_topic_embeddings, run_mingpao_pipeline, run_oriental_pipeline, run_hkgov_pipeline

st.set_page_config(page_title="News Topic Retriever", layout="wide", page_icon="📰",
    menu_items={
        'About': 'GitHub: https://github.com/phykawing'
    })

st.title("📰 News Topic Retriever")
st.markdown("Upload labeled news samples to retrieve similar articles from Mingpao, Oriental Daily, and HK Gov News.")

uploaded_file = st.file_uploader("📂 Upload `NewsDataSave.csv`", type="csv", on_change=lambda: st.session_state.pop("topic_vectors", None))

threshold = st.slider("🔍 Similarity Threshold", 0.1, 1.0, 0.7, step=0.05)

def display_results(results_df):
    if results_df.empty:
        st.warning("No matching articles found.")
        return

    for cat in results_df.Topic.unique():
        st.markdown(f"### Category: {cat}")
        cat_results = results_df[results_df['Topic'] == cat]
        for _, row in cat_results.iterrows():
            st.markdown(f"**[{row['Titles']}]({row['Links']})**")
            st.markdown(f"Similarity: {row['Similarity']}")
            # st.markdown("---")

if uploaded_file:
    df_labeled = pd.read_csv(uploaded_file)
    if "topic_vectors" not in st.session_state:
        with st.spinner("Loading Topic Vectors..."):
            st.session_state["topic_vectors"] = generate_topic_embeddings(df_labeled)
    topic_vectors = st.session_state["topic_vectors"]
    st.success(f"Labeled Data Loaded!")
    st.markdown("### Categories:")
    for cat in df_labeled.Categories.unique():
        st.markdown(f"{cat}")
    st.button("🔄 Refresh Topic Vectors", on_click=lambda: st.session_state.pop("topic_vectors", None))

    if st.button("🚀 Run News Retrieval"):
        with st.spinner("Processing Mingpao..."):
            results_df = run_mingpao_pipeline(topic_vectors, similarity_threshold=threshold)
        st.success(f"Retrieved {len(results_df)} matching articles!")
        st.markdown("# Mingpao Results:")
        display_results(results_df)
        st.markdown("---")

        with st.spinner("Processing Oriental Daily..."):
            results_df = run_oriental_pipeline(topic_vectors, similarity_threshold=threshold)
        st.success(f"Retrieved {len(results_df)} matching articles!")
        st.markdown("# Oriental Daily Results:")
        display_results(results_df)
        st.markdown("---")

        with st.spinner("Processing HK Gov News..."):
            results_df = run_hkgov_pipeline(topic_vectors, similarity_threshold=threshold)
        st.success(f"Retrieved {len(results_df)} matching articles!")
        st.markdown("# HK Gov News Results:")
        display_results(results_df)
        st.markdown("---")