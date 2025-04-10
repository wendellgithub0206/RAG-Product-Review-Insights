import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_cleaned_df(df, save_path="./eda_output"):
    os.makedirs(save_path, exist_ok=True)
    plt.rcParams['font.family'] = 'Microsoft JhengHei'
    print("資料視覺化分析")


    #1. 情感標籤分佈（label: 0=負評, 1=中立, 2=正評）
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label_counts = df["label"].value_counts().sort_index()
    label_names = [label_map[i] for i in label_counts.index]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=label_names, y=label_counts.values, palette="Set2")
    plt.title("情感分佈")
    plt.ylabel("評論數")
    plt.savefig(f"{save_path}/sentiment_distribution.png")
    plt.close()

    #2. 熱門產品統計
    top_products = df["product_title"].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
    plt.title("熱門產品（出現次數）")
    plt.xlabel("評論數")
    plt.savefig(f"{save_path}/top_products.png")
    plt.close()

    #4. 類別統計
    category_counts = df["category"].value_counts()

    plt.figure(figsize=(6, 4))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette="pastel")
    plt.title("主分類統計")
    plt.ylabel("評論數")
    plt.savefig(f"{save_path}/category_distribution.png")
    plt.close()

    # 5. 打印文字統計
    print("EDA 統計摘要")
    print(f"總評論數: {len(df)}")
    print(f"平均字數: {df['text_length'].mean():.2f}")
    print(f"正評: {label_counts.get(2, 0)}")
    print(f"中立: {label_counts.get(1, 0)}")
    print(f"負評: {label_counts.get(0, 0)}")
    print(f"分類：{category_counts.to_dict()}")

    print(f"\n分析圖儲存至：{save_path}/")

