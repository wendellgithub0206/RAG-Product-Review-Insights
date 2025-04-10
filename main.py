from datasets import load_from_disk
from transformers import AutoTokenizer
import pandas as pd
from model import train_and_save_model
from inference import load_model_and_predict
from analysis import analyze_cleaned_df

def main():
    data_dir = "./data/tokenized_amazon_sentiment"
    model_dir = "./model_output"

    #1. 載入儲存的 tokenized dataset 和 tokenizer
    print("載入預處理資料與tokenizer")
    tokenized_data = load_from_disk(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(data_dir)

    #2. 載入清洗過的原始資料，供分析使用
    print("資料分析")
    df = pd.read_csv(f"{data_dir}/cleaned_reviews_sentiment.csv")
    analyze_cleaned_df(df)

    #3. 訓練情感分析模型
    print("訓練模型")
    tokenizer = train_and_save_model(
        tokenized_dataset=tokenized_data,
        tokenizer=tokenizer,
        output_dir=model_dir,
        num_epochs=4,
        batch_size=16,
        learning_rate=2e-5,
        early_stopping_patience=2
    )

    #4. 測試推論
    print("測試模型推論")
    test_samples = [
        "This product is absolutely fantastic. It exceeded all my expectations!", #正面
        "The quality is average, not really what I was expecting based on the description.",#中立
        "Terrible product. It broke after one day of use. Don't waste your money."#負面
    ]

    for text in test_samples:
        result = load_model_and_predict(
            text=text,
            model_path=f"{model_dir}/final_model"
        )
        print(f"\n文本: {result['text']}")
        print(f"預測情感: {result['predicted_sentiment']} (置信度: {result['confidence']:.4f})")

if __name__ == "__main__":
    main()
