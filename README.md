# 📰 News Topic Retriever using Embedding Similarity

This Jupyter notebook provides a pipeline for retrieving and classifying news articles based on user-defined topics using embedding similarity. It is particularly designed for Traditional Chinese content and leverages both RSS feeds and web scraping to source the news.

# 📰 使用嵌入相似度的新聞主題檢索工具

本 Jupyter Notebook 提供一個基於使用者定義主題，透過嵌入向量相似度來檢索並分類新聞文章的流程。此工具特別適用於繁體中文內容，並結合 RSS 及網頁爬蟲技術來擷取新聞資料。



## 🛠️ Technologies Used

- `numpy`, `pandas` – numerical and tabular data processing.
- `BeautifulSoup` – HTML parsing for web scraping.
- `langchain` – embedding and LLM framework.
- `FAISS` – efficient similarity search for dense vectors.
- `BAAI/bge-m3` – embedding model for Traditional Chinese.

## 🛠️ 使用技術

- `numpy`, `pandas` – 數據處理與分析。
- `BeautifulSoup` – 用於 HTML 解析與網頁擷取。
- `langchain` – 向量與語言模型框架。
- `FAISS` – 高效的密集向量相似度搜尋工具。
- `BAAI/bge-m3` – 適用於繁體中文的語意嵌入模型。



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

## 📌 功能概覽

本 Notebook 執行以下關鍵操作：

1. **主題定義**：使用標記過的資料集（`NewsDataSave.csv`），其中新聞範例依主題分類，並標記為正樣本或負樣本。
2. **嵌入向量生成**：  
   - 根據標記範例為每個主題建立嵌入向量。
   - 使用適用於繁體中文的 `BAAI/bge-m3` 模型來產生嵌入。
   - 主題嵌入向量透過正樣本向量的加總減去負樣本向量的加總來建立。
3. **新聞擷取**：
   - 從明報、東方日報及香港政府新聞等來源實時擷取新聞。
   - 對擷取的文章建立嵌入並與主題向量進行比對。
4. **相似度比對**：
   - 使用 FAISS 執行餘弦相似度比較來評估新聞與主題的關聯性。
   - 篩選出超過指定相似度門檻的新聞。



## 📁 Input Dataset

**File**: `NewsDataSave.csv`  
**Columns**:
- `Categories`: Topic name.
- `Negative`: `1` = Negative sample, `0` = Positive sample.
- `Titles`: News headline.
- `Links`: Source URL.
- `Summary`: Full or partial news content.

## 📁 輸入資料集

**檔案**：`NewsDataSave.csv`  
**欄位說明**：
- `Categories`：主題名稱
- `Negative`：`1` 表示負樣本，`0` 表示正樣本
- `Titles`：新聞標題
- `Links`：新聞來源網址
- `Summary`：新聞內文或摘要



## ⚙️ Embedding Strategy

- Each news sample’s embedding = `embedding(Title + Summary)`.

## ⚙️ 嵌入策略

- 每則新聞樣本的嵌入 = `embedding(Title + Summary)`



## 🧠 Embedding Model

- **Default**: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) — optimized for Traditional Chinese text.

## 🧠 嵌入模型

- **預設模型**：[`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) —— 為繁體中文優化的語言模型。



## 🔍 Similarity Search

- **Library**: [FAISS](https://github.com/facebookresearch/faiss)
- Efficient approximate nearest neighbor (ANN) search on high-dimensional vectors.

## 🔍 相似度搜尋

- **使用套件**：[FAISS](https://github.com/facebookresearch/faiss)  
- 高效的高維向量近似最近鄰搜尋工具。



## 🌐 News Sources

Example sources integrated via RSS and scraping:
- [Mingpao](https://news.mingpao.com/pns/%E6%AF%8F%E6%97%A5%E6%98%8E%E5%A0%B1/main)
- [Oriental Daily](https://orientaldaily.on.cc/)
- [Hong Kong Government News](https://www.info.gov.hk/gia/general/today.htm)

## 🌐 新聞來源

透過 RSS 與爬蟲技術整合的新聞來源包括：
- [明報](https://news.mingpao.com/pns/%E6%AF%8F%E6%97%A5%E6%98%8E%E5%A0%B1/main)
- [東方日報](https://orientaldaily.on.cc/)
- [香港政府新聞網](https://www.info.gov.hk/gia/general/ctoday.htm)



## 🚀 Usage

1. Update or customize `NewsDataSave.csv` with your own topics and samples.
2. Run the notebook step by step to:
   - Generate topic embeddings.
   - Retrieve live news.
   - Compute similarity.
   - View the filtered list of matched articles.

## 🚀 使用方式

1. 根據需求更新或自訂 `NewsDataSave.csv` 中的主題與樣本。
2. 依序執行 Notebook 步驟以：
   - 產生主題嵌入向量
   - 擷取即時新聞
   - 計算相似度
   - 查看符合條件的新聞清單



## 🛠️ Customization

- Change embedding models via Hugging Face Transformers.
- Adjust FAISS index parameters for performance tuning.
- Modify similarity threshold as per precision/recall trade-off.

## 🛠️ 客製化選項

- 可透過 Hugging Face 調整使用的嵌入模型。
- 可自訂 FAISS 索引參數以調整效能。
- 可調整相似度門檻，以控制檢索的精準度與召回率。



## 📎 Notes

- Ensure internet access for RSS and web scraping.
- For large datasets or frequent scraping, consider caching or batch processing.
- Ensure compliance with each news site's usage policy when scraping.

## 📎 注意事項

- 執行需具備網路連線，以擷取即時新聞。
- 若資料量龐大，建議加上快取或分批處理機制。
- 擷取資料請遵守各新聞網站的使用政策與版權規定。


## 💻 Streamlit App Interface

In addition to the Jupyter notebook, this project includes a Streamlit web app for interactive use.

## 💻 Streamlit 應用介面

除了 Jupyter Notebook，本專案亦包含一個基於 Streamlit 的網頁應用程式，供互動使用。

### 📂 Files
- `app.py` – Streamlit-based UI for uploading labeled datasets and retrieving matching news.
- `news_pipeline.py` – Core logic for embedding generation and news retrieval pipelines.

### 📂 檔案說明

- `app.py` – 使用 Streamlit 架設的介面，用於上傳標記資料集並檢索相似新聞。
- `news_pipeline.py` – 負責嵌入生成與新聞檢索流程的核心邏輯。

### ▶️ Running the App

Make sure all dependencies are installed (including `streamlit`, `langchain`, `huggingface_hub`, `faiss`, `bs4`, etc.).

Then, launch the app with:

```bash
streamlit run app.py
```

### ▶️ 執行應用程式

請先確保已安裝所有必要套件（包括 `streamlit`、`langchain`、`huggingface_hub`、`faiss`、`bs4` 等）。

接著使用以下指令啟動應用程式：

```bash
streamlit run app.py
```

### 🧭 Functionality

- Upload your `NewsDataSave.csv` via the UI.
- Adjust similarity threshold with a slider.
- Click “Run News Retrieval” to:
  - Generate topic embeddings.
  - Fetch and embed articles from:
    - Mingpao
    - Oriental Daily
    - HK Government News
  - Match and display articles exceeding the similarity threshold.

### 🧭 功能說明

- 透過網頁介面上傳 `NewsDataSave.csv`。
- 可使用滑桿調整相似度門檻。
- 點擊「Run News Retrieval（執行新聞檢索）」按鈕，即可：
  - 產生主題嵌入向量。
  - 擷取並嵌入下列來源的新聞：
    - 明報
    - 東方日報
    - 香港政府新聞
  - 篩選並顯示超過指定相似度門檻的新聞文章。

