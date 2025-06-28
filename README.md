# 📰 News Topic Retriever using Embedding Similarity
# 📰 使用嵌入相似度的新聞主題檢索工具


This Jupyter notebook provides a pipeline for retrieving and classifying news articles based on user-defined topics using embedding similarity. It is particularly designed for Traditional Chinese content and leverages both RSS feeds and web scraping to source the news.
本 Jupyter Notebook 提供一個基於使用者定義主題，透過嵌入向量相似度來檢索並分類新聞文章的流程。此工具特別適用於繁體中文內容，並結合 RSS 及網頁爬蟲技術來擷取新聞資料。


## 🛠️ Technologies Used
## 🛠️ 使用技術

- `numpy`, `pandas` – numerical and tabular data processing.
- `numpy`, `pandas` – 數據處理與分析。
- `BeautifulSoup` – HTML parsing for web scraping.
- `BeautifulSoup` – 用於 HTML 解析與網頁擷取。
- `langchain` – embedding and LLM framework.
- `langchain` – 向量與語言模型框架。
- `FAISS` – efficient similarity search for dense vectors.
- `FAISS` – 高效的密集向量相似度搜尋工具。
- `BAAI/bge-m3` – embedding model for Traditional Chinese.
- `BAAI/bge-m3` – 適用於繁體中文的語意嵌入模型。

## 📌 Overview
## 📌 功能概覽


The notebook performs the following key operations:
本 Notebook 執行以下關鍵操作：


1. **Topic Definition**: Uses a labeled dataset (`NewsDataSave.csv`) where news examples are categorized by topic, labeled as positive or negative samples.
1. **主題定義**：使用標記過的資料集（`NewsDataSave.csv`），其中新聞範例依主題分類，並標記為正樣本或負樣本。
2. **Embedding Generation**: 
2. **嵌入向量生成**：  
   - Constructs embedding vectors for each topic based on labeled examples.
   - 根據標記範例為每個主題建立嵌入向量。
   - Embeddings are calculated using the `BAAI/bge-m3` model for Traditional Chinese.
   - 使用適用於繁體中文的 `BAAI/bge-m3` 模型來產生嵌入。
   - Topic embeddings are computed by subtracting the aggregated negative sample vectors from the positive ones.
   - 主題嵌入向量透過正樣本向量的加總減去負樣本向量的加總來建立。
3. **News Retrieval**:
3. **新聞擷取**：
   - Collects real-time news from selected sources: Mingpao, Oriental Daily, and Hong Kong Government News.
   - 從明報、東方日報及香港政府新聞等來源實時擷取新聞。
   - Generates embeddings for the retrieved articles and compares them to the topic vectors.
   - 對擷取的文章建立嵌入並與主題向量進行比對。
4. **Similarity Matching**:
4. **相似度比對**：
   - Uses cosine similarity (via FAISS) to determine news-topic relevance.
   - 使用 FAISS 執行餘弦相似度比較來評估新聞與主題的關聯性。
   - Filters and lists articles that exceed a specified similarity threshold.
   - 篩選出超過指定相似度門檻的新聞。


## 📁 Input Dataset
## 📁 輸入資料集


**File**: `NewsDataSave.csv`  
**檔案**：`NewsDataSave.csv`  
**Columns**:
**欄位說明**：
- `Categories`: Topic name.
- `Categories`：主題名稱
- `Negative`: `1` = Negative sample, `0` = Positive sample.
- `Negative`：`1` 表示負樣本，`0` 表示正樣本
- `Titles`: News headline.
- `Titles`：新聞標題
- `Links`: Source URL.
- `Links`：新聞來源網址
- `Summary`: Full or partial news content.
- `Summary`：新聞內文或摘要


## ⚙️ Embedding Strategy
## ⚙️ 嵌入策略


- Each news sample’s embedding = `embedding(Title + Summary)`.
- 每則新聞樣本的嵌入 = `embedding(Title + Summary)`
- Each topic’s embedding vector is derived as:
- 每個主題的嵌入向量計算方式：
  

  ```
  ```
  normalized(
  normalized(
    mean(positive_example_vectors) - mean(negative_example_vectors)
    mean(positive_example_vectors) - mean(negative_example_vectors)
  )
  )
  ```
  ```


## 🧠 Embedding Model
## 🧠 嵌入模型


- **Default**: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) — optimized for Traditional Chinese text.
- **預設模型**：[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) —— 為繁體中文優化的語言模型。


## 🔍 Similarity Search
## 🔍 相似度搜尋


- **Library**: [FAISS](https://github.com/facebookresearch/faiss)
- **使用套件**：[FAISS](https://github.com/facebookresearch/faiss)  
- Efficient approximate nearest neighbor (ANN) search on high-dimensional vectors.
- 高效的高維向量近似最近鄰搜尋工具。


## 🌐 News Sources
## 🌐 新聞來源


Example sources integrated via RSS and scraping:
透過 RSS 與爬蟲技術整合的新聞來源包括：
- Mingpao
- 明報
- Oriental Daily
- 東方日報
- Hong Kong Government News
- 香港政府新聞網


## 🚀 Usage
## 🚀 使用方式


1. Update or customize `NewsDataSave.csv` with your own topics and samples.
1. 根據需求更新或自訂 `NewsDataSave.csv` 中的主題與樣本。
2. Run the notebook step by step to:
2. 依序執行 Notebook 步驟以：
   - Generate topic embeddings.
   - 產生主題嵌入向量
   - Retrieve live news.
   - 擷取即時新聞
   - Compute similarity.
   - 計算相似度
   - View the filtered list of matched articles.
   - 查看符合條件的新聞清單


## 🛠️ Customization
## 🛠️ 客製化選項


- Change embedding models via Hugging Face Transformers.
- 可透過 Hugging Face 調整使用的嵌入模型。
- Adjust FAISS index parameters for performance tuning.
- 可自訂 FAISS 索引參數以調整效能。
- Modify similarity threshold as per precision/recall trade-off.
- 可調整相似度門檻，以控制檢索的精準度與召回率。

## 📎 Notes
## 📎 注意事項

- Ensure internet access for RSS and web scraping.
- 執行需具備網路連線，以擷取即時新聞。
- For large datasets or frequent scraping, consider caching or batch processing.
- 若資料量龐大，建議加上快取或分批處理機制。
- Ensure compliance with each news site's usage policy when scraping.
- 擷取資料請遵守各新聞網站的使用政策與版權規定。

