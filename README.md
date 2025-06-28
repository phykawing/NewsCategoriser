# 📰 News Topic Retriever using Embedding Similarity


This Jupyter notebook provides a pipeline for retrieving and classifying news articles based on user-defined topics using embedding similarity. It is particularly designed for Traditional Chinese content and leverages both RSS feeds and web scraping to source the news.


## 🛠️ Technologies Used

- `numpy`, `pandas` – numerical and tabular data processing.
- `BeautifulSoup` – HTML parsing for web scraping.
- `langchain` – embedding and LLM framework.
- `FAISS` – efficient similarity search for dense vectors.
- `BAAI/bge-m3` – embedding model for Traditional Chinese.

## 📌 Overview


The notebook performs the following key operations:


1. **Topic Definition**: Uses a labeled dataset (`NewsDataSave.csv`) where news examples are categorized by topic, labeled as positive or negative samples.
2. **Embedding Generation**: 
   - Constructs embedding vectors for each topic based on labeled examples.
   - Embeddings are calculated using the `BAAI/bge-m3` model for Traditional Chinese.
   - Topic embeddings are computed by subtracting the aggregated negative sample vectors from the positive ones.
3. **News Retrieval**:
   - Collects real-time news from selected sources: Mingpao, Oriental Daily, and Hong Kong Government News.
   - Generates embeddings for the retrieved articles and compares them to the topic vectors.
4. **Similarity Matching**:
   - Uses cosine similarity (via FAISS) to determine news-topic relevance.
   - Filters and lists articles that exceed a specified similarity threshold.


## 📁 Input Dataset


**File**: `NewsDataSave.csv`  
**Columns**:
- `Categories`: Topic name.
- `Negative`: `1` = Negative sample, `0` = Positive sample.
- `Titles`: News headline.
- `Links`: Source URL.
- `Summary`: Full or partial news content.


## ⚙️ Embedding Strategy


- Each news sample’s embedding = `embedding(Title + Summary)`.



## 🧠 Embedding Model


- **Default**: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) — optimized for Traditional Chinese text.

## 🔍 Similarity Search

- **Library**: [FAISS](https://github.com/facebookresearch/faiss)
- Efficient approximate nearest neighbor (ANN) search on high-dimensional vectors.


## 🌐 News Sources


Example sources integrated via RSS and scraping:
- Mingpao
- Oriental Daily
- Hong Kong Government News


## 🚀 Usage


1. Update or customize `NewsDataSave.csv` with your own topics and samples.
2. Run the notebook step by step to:
   - Generate topic embeddings.
   - Retrieve live news.
   - Compute similarity.
   - View the filtered list of matched articles.


## 🛠️ Customization


- Change embedding models via Hugging Face Transformers.
- Adjust FAISS index parameters for performance tuning.
- Modify similarity threshold as per precision/recall trade-off.

## 📎 Notes

- Ensure internet access for RSS and web scraping.
- For large datasets or frequent scraping, consider caching or batch processing.
- Ensure compliance with each news site's usage policy when scraping.

