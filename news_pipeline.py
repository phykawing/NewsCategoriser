import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document as LangchainDocument
import numpy as np
import time
import requests

# Initialize embedding model
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # True: Normalize for cosine similarity
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def generate_topic_embeddings(mdata):
    ev_cat = {}
    # Generate embedding vectors for each category based on labeled data
    for cat in mdata.Categories.unique():

        print(cat)

        ev_list = []

        for neg_sample in [0, 1]:

            if neg_sample == 0:
                print("Positive Sample...")
            else:
                print("Negative Sample...")

            sample_list = []

            for url in mdata.loc[mdata.Categories == cat].loc[mdata.Negative == neg_sample]['Links']:

                if mdata.loc[mdata.Categories == cat].loc[mdata.Negative == neg_sample].loc[mdata.Links == url]['Summary'].values[0] == "":

                    raise ValueError("No text summary!")

                else:
                    title = mdata.loc[mdata.Categories == cat].loc[mdata.Negative == neg_sample].loc[mdata.Links == url]['Titles'].values[0]
                    content = mdata.loc[mdata.Categories == cat].loc[mdata.Negative == neg_sample].loc[mdata.Links == url]['Summary'].values[0]

                # Create embeddings using title and content
                sample_list.append(embeddings.embed_query(title + ' ' + content))
                # print(title + ' ' + content)

            ev = np.sum(sample_list, axis=0)

            ev_list.append(ev)

        ev_avg = ev_list[0] - ev_list[1]  # Positive - Negative
        ev_avg = ev_avg / np.linalg.norm(ev_avg)  # Normalize the vector

        ev_cat[cat] = ev_avg
    return ev_cat

def vector_search_by_vector(db, topic_vectors, k=15, score_threshold=0.7):
    """
    Perform vector search on the FAISS database.
    """
    results = []
    for topic, vector in topic_vectors.items():
        docs_and_scores = db.similarity_search_with_score_by_vector(vector, k=15, score_threshold=score_threshold)
        for doc, score in docs_and_scores:
            results.append({
                "Topic": topic,
                "Titles": doc.metadata["title"],
                "Links": doc.metadata["source"],
                "Similarity": score
            })
    return pd.DataFrame(results)

def fetch_mingpao_news():
    print("Fetching Mingpao news...")
    rss_list = ["https://news.mingpao.com/rss/pns/s00001.xml", #要聞
                "https://news.mingpao.com/rss/pns/s00002.xml", #港聞
                "https://news.mingpao.com/rss/pns/s00003.xml", #社評
                "https://news.mingpao.com/rss/pns/s00004.xml", #經濟
                "https://news.mingpao.com/rss/pns/s00005.xml", #副刊
                "https://news.mingpao.com/rss/pns/s00011.xml", #教育
                "https://news.mingpao.com/rss/pns/s00012.xml", #觀點
                "https://news.mingpao.com/rss/pns/s00013.xml", #中國
                "https://news.mingpao.com/rss/pns/s00014.xml", #國際
                "https://news.mingpao.com/rss/pns/s00015.xml", #體育
                "https://news.mingpao.com/rss/pns/s00016.xml", #娛樂
                "https://news.mingpao.com/rss/pns/s00017.xml", #英文
                "https://news.mingpao.com/rss/pns/s00018.xml" #作家專欄
                ]
    feed_list = []
    for rss in rss_list:
        feed = feedparser.parse(rss)
        feed_list += feed.entries
    today = datetime.now(timezone(timedelta(hours=8)))
    today = today.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone(timedelta(hours=0)))
    feed_list = [feed for feed in feed_list if date_parser.parse(feed.published) >= today]
    print('No. of news:', len(feed_list))
    return feed_list

def run_mingpao_pipeline(topic_vectors, similarity_threshold=0.7):
    feed_list = fetch_mingpao_news()

    documents = []
    for entry in feed_list:
        content = entry.title + " " + entry.summary
        metadata = {
            "source": "/".join(entry.link.split("/")[:-1]),
            "title": entry.title,
            "newspaper": "明報"
        }
        documents.append(LangchainDocument(page_content=content, metadata=metadata))

    # Embed news and load into FAISS
    db = FAISS.from_documents(documents, embeddings, distance_strategy=DistanceStrategy.COSINE) #EUCLIDEAN_DISTANCE/MAX_INNER_PRODUCT/DOT_PRODUCT/JACCARD/COSINE

    print(db.distance_strategy)

    # Perform vector search
    results = vector_search_by_vector(db, topic_vectors, k=15, score_threshold=similarity_threshold)

    return pd.DataFrame(results)


def fetch_oriental_news():
    print("Fetching Oriental Daily news...")
    today = datetime.now()
    url = r'https://orientaldaily.on.cc/section/sitemap/' + today.date().strftime('%Y%m%d')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    items = soup.find_all('li', attrs={'class': 'item'})

    on_list = []

    for item in items:
        title = item.text
        link = r'https://orientaldaily.on.cc' + item.find('a')['href']
        try:
            response = requests.get(link)
        except:
            time.sleep(3)
            response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        paras = soup.find_all('div', class_ = 'paragraph')

        content = ''

        for p in paras:
            content += p.get_text()

        content = ''.join(content.split())

        on_list.append([title, link, content])

    print('No. of news:', len(on_list))
    return on_list

def run_oriental_pipeline(topic_vectors, similarity_threshold=0.7):
    on_list = fetch_oriental_news()

    on_documents = []
    for entry in on_list:
        content = entry[0] + ' ' + entry[2]
        metadata = {
            "source": "/".join(entry[1].split("/")[:-1]) + "/",
            "title": entry[0],
            "newspaper": "東方日報"
        }
        on_documents.append(LangchainDocument(page_content=content, metadata=metadata))

    # Embed news and load into FAISS
    db = FAISS.from_documents(on_documents, embeddings, distance_strategy=DistanceStrategy.COSINE) #EUCLIDEAN_DISTANCE/MAX_INNER_PRODUCT/DOT_PRODUCT/JACCARD/COSINE

    print(db.distance_strategy)

    # Perform vector search
    results = vector_search_by_vector(db, topic_vectors, k=15, score_threshold=similarity_threshold)

    return pd.DataFrame(results)

def fetch_hkgov_news():
    print("Fetching HK Gov news...")
    rss_list_hkgov = ["https://www.info.gov.hk/gia/rss/general_zh.xml"] #新聞公報

    feed_list_hkgov = []

    # Parse HK Gov RSS feeds
    for rss in rss_list_hkgov:
        feed = feedparser.parse(rss)
        feed_list_hkgov += feed.entries

    yesterday = datetime.now(timezone(timedelta(hours=8))) - timedelta(hours=24)
    print("Yesterday's date:", yesterday.date())
    yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone(timedelta(hours=0)))


    feed_list_hkgov = [feed for feed in feed_list_hkgov if date_parser.parse(feed.published) >= yesterday]
    
    print('No. of news:', len(feed_list_hkgov))
    return feed_list_hkgov

def run_hkgov_pipeline(topic_vectors, similarity_threshold=0.7):
    hkgov_list = fetch_hkgov_news()

    hkgov_documents = []
    for entry in hkgov_list:
        soup = BeautifulSoup(entry.summary, 'html.parser')
        entry_summary = soup.text.replace("\n", "")

        content = entry.title + ' ' + entry_summary
        metadata = {
            "source": entry.link,
            "title": entry.title,
            "newspaper": "新聞公報"
        }
        hkgov_documents.append(LangchainDocument(page_content=content, metadata=metadata))

    # Embed news and load into FAISS
    db = FAISS.from_documents(hkgov_documents, embeddings, distance_strategy=DistanceStrategy.COSINE) #EUCLIDEAN_DISTANCE/MAX_INNER_PRODUCT/DOT_PRODUCT/JACCARD/COSINE

    print(db.distance_strategy)

    # Perform vector search
    results = vector_search_by_vector(db, topic_vectors, k=15, score_threshold=0.7)

    return pd.DataFrame(results)