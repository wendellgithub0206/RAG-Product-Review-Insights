import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import defaultdict

def label(rating):
    # 根據評分標記情感類別
    if rating <= 2:
        return "Negative"  # 負面
    elif rating == 3:
        return "Neutral"  # 中立
    else:
        return "Positive"  # 正面
def label2id(df):
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df["label"] = df["label"].map(label_map)
    return df

def load_amazon_reviews(categories=("Appliances", "Electronics"), sample_size=80000):
    all_reviews = []
    for category in categories:
        print(f"載入 {category} ")
        split_str = f"full[:{sample_size}]" if sample_size else "full"

        reviews = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split=split_str,
            trust_remote_code=True
        )
        meta = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True
        )

        df_reviews = reviews.to_pandas()
        df_meta = meta.to_pandas()

        df_reviews["category"] = category
        df_merged = pd.merge(df_reviews, df_meta, how="left", on="parent_asin")
        all_reviews.append(df_merged)

    df = pd.concat(all_reviews, ignore_index=True)
    df = df[["text", "rating", "title_x", "title_y", "parent_asin", "category", "categories"]].dropna()
    df = df[df["text"].str.len() > 20]
    df["label"] = df["rating"].apply(label)

    df = df.rename(columns={
        "text": "review_text",
        "title_x": "review_title",
        "title_y": "product_title"
    })

    return df

def save_tokenized(df, model_name="bert-base-uncased", save_dir="./data/tokenized_amazon_sentiment"):
    print("分割train/test...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(examples):
        return tokenizer(examples["review_text"], truncation=True, padding="max_length", max_length=256)

    print("Tokenizing train set...")
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.remove_columns(set(train_dataset.column_names) - {"input_ids", "attention_mask", "label"})

    print("Tokenizing test set...")
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.remove_columns(set(test_dataset.column_names) - {"input_ids", "attention_mask", "label"})

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    os.makedirs(save_dir, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 儲存資料
    df.to_csv(os.path.join(save_dir, "reviews_sentiment.csv"), index=False)

    print(f"所有資料儲存完成：{save_dir}")

def save_rag(df, save_dir="./data/tokenized_amazon_rag"):
    # 儲存商品的評論資料（不包括情感標籤）
    df_rag = df[["parent_asin", "product_title", "category", "review_text"]]
    
    # 儲存原始評論資料
    os.makedirs(save_dir, exist_ok=True)
    df_rag.to_csv(os.path.join(save_dir, "rag_data.csv"), index=False)
    
    print(f"RAG 資料儲存完成：{save_dir}")

def labels_downsampling(df):
    # 進行類別下採樣
    label_counts = df["label"].value_counts()
    min_count = label_counts.min()

    print(f"原始樣本分佈：\n{label_counts}")
    print(f"將所有類別下採樣至 {min_count} 筆")

    balanced_df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

    print(f"下採樣後樣本分佈：\n{balanced_df['label'].value_counts()}")
    return balanced_df

if __name__ == "__main__":
    #1.讀取資料
    df = load_amazon_reviews(sample_size=80000)
    #2.label轉id
    df = label2id(df)
    #3. 先儲存RAG資料集（原始評論，不合併）才不會因為下採樣減少資料
    save_rag(df)
    #下採樣
    df = labels_downsampling(df)
    #4. 儲存情緒分類資料集
    save_tokenized(df)

  
    