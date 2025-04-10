# build_index.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm import tqdm

def create_faiss_index(data_path="./data/tokenized_amazon_rag/rag_data.csv", save_dir="./rag"):
    # 讀取資料
    print("讀取RAG資料")
    df = pd.read_csv(data_path)
    
    # 過濾太短的評論
    df = df[df["review_text"].str.len() > 20]
    
    # 為評論文本添加商品信息，方便後續識別
    df["combined_text"] = df["product_title"] + " - " + df["review_text"]
    
    # 初始化模型 使用這個模型支援中文
    print("初始化向量模型")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # 將所有評論轉換為嵌入向量
    print("生成評論的嵌入向量")
    embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)

    # 創建 FAISS 索引
    print("建立 FAISS 向量索引")
    dimension = embeddings.shape[1]  # 向量的維度
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # 添加向量到索引中

    # 儲存索引
    os.makedirs(save_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(save_dir, "faiss_index.index"))

    # 創建和儲存原始評論的對應信息
    source_texts = []
    for _, row in df.iterrows():
        source_texts.append({
            "parent_asin": row["parent_asin"],
            "product_title": row["product_title"],
            "category": row["category"],
            "review_text": row["review_text"],
            "combined_text": row["combined_text"]
        })
    
    with open(os.path.join(save_dir, "source_texts.pkl"), "wb") as f:
        pickle.dump(source_texts, f)

    print(f"索引建立並儲存至：{save_dir}")
    print(f"總共索引了 {len(source_texts)} 筆評論")

if __name__ == "__main__":
    create_faiss_index()