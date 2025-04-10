import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
from tqdm import tqdm

def load_model_and_predict(
    model_path="./model_output/final_model",
    text=None,
    csv_path=None,
    max_length=256
):
    print(f"從 {model_path} 載入模型與 tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        return_all_scores=True
    )

    # 從模型自動取得 id2label
    id2label = model.config.id2label = { 0: "負評", 1: "中立", 2: "正評"}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    if text:
        return predict_sentiment(text, sentiment_analyzer, tokenizer, max_length, id2label)
    elif csv_path:
        return batch_predict(csv_path, sentiment_analyzer, tokenizer, max_length, id2label)
    else:
        print("錯誤: 請提供 text 或 csv_path")
        return None

def predict_sentiment(text, sentiment_analyzer, tokenizer, max_length, id2label):
    # 截斷長文本
    if len(tokenizer.encode(text)) > max_length:
        encoded = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
        text = tokenizer.decode(encoded, skip_special_tokens=True)

    results = sentiment_analyzer(text)[0]  # 取出第一組結果
    scores = {}

    for score in results:
        try:
            label_id = int(score["label"].split("_")[-1])
        except ValueError:
            label_id = [k for k, v in id2label.items() if v.lower() in score["label"].lower()][0]
        label_name = id2label[label_id]
        scores[label_name] = score["score"]

    predicted_label = max(scores, key=scores.get)

    return {
        "text": text,
        "predicted_sentiment": predicted_label,
        "confidence": scores[predicted_label],
        "scores": scores
    }

def batch_predict(csv_path, sentiment_analyzer, tokenizer, max_length, id2label):
    df = pd.read_csv(csv_path, low_memory=False)

    if "reviews.text" in df.columns:
        text_column = "reviews.text"
    elif "text" in df.columns:
        text_column = "text"
    else:
        print("找不到評論欄位")
        return None

    if "name" in df.columns or "product_name" in df.columns:
        name_column = "name" if "name" in df.columns else "product_name"
        df["full_text"] = df[name_column] + " - " + df[text_column]
        texts = df["full_text"].tolist()
    else:
        texts = df[text_column].tolist()

    print(f"共 {len(texts)} 筆...")
    results = []

    for text in tqdm(texts, desc="處理中"):
        result = predict_sentiment(text, sentiment_analyzer, tokenizer, max_length, id2label)
        results.append(result)

    result_df = pd.DataFrame(results)
    output_df = pd.concat([df.reset_index(drop=True), result_df[["predicted_sentiment", "confidence"]]], axis=1)

    output_path = csv_path.replace(".csv", "_with_predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"批次預測已儲存至: {output_path}")

    return output_df

