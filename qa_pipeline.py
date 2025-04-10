from rag_utils import RAGRetriever
from inference import load_model_and_predict
from openai import OpenAI
from collections import defaultdict
import os
from dotenv import load_dotenv

#載入.env
load_dotenv()

#初始化 GPT 客戶端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_sentiments(reviews, model_path="./model_output/final_model"):
    #對每個商品的評論進行情感分析
    print("執行情感分析...")
    results = []
    
    for item in reviews:
        product = item["product"]
        category = item["category"]
        
        # 遍歷該商品的所有評論（reviews是列表）
        for review in item["reviews"]:   
            result = load_model_and_predict(text=review, model_path=model_path)
            results.append({
                "text": review,
                "label": result["predicted_sentiment"],
                "product": product,
                "category": category
            })
    
    return results

def group_reviews_by_product_and_sentiment(labeled_reviews):
    #將評論按產品和情感分組
    product_sentiments = defaultdict(lambda: {"正評": [], "中立": [], "負評": []})
    stats_summary = defaultdict(lambda: {"正評": 0, "中立": 0, "負評": 0})

    for item in labeled_reviews:
        review = item["text"]
        label = item["label"]
        product = item["product"]

        product_sentiments[product][label].append(review)
        stats_summary[product][label] += 1

    return product_sentiments, stats_summary

def summarize_reviews_by_product(product_reviews_dict):
    """使用GPT為每個產品的不同情感評論生成摘要"""
    product_summaries = {}
    for product, sentiments in product_reviews_dict.items():
        product_summaries[product] = {}
        
        for label, reviews in sentiments.items():
            if not reviews:
                continue

            # 只取前max_reviews條評論用於總結
            max_reviews = 10
            reviews_to_summarize = reviews[:max_reviews]
            joined = "\n".join(reviews_to_summarize)
            
            prompt = f"""
以下是產品「{product}」的「{label}」評論，請總結出用戶提到的前3個主要觀點：

{joined}

請用簡潔的條列式繁體中文回答，每點控制在20字以內。格式：
- 第一點觀點
- 第二點觀點
- 第三點觀點
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=250
                )
                product_summaries[product][label] = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"調用GPT API時出錯: {e}")
                product_summaries[product][label] = f"無法總結 {label} 評論"
                
    return product_summaries

def summarize_reviews(question: str):
    #主要處理流程：檢索 -> 情感分析 -> 分組 -> 摘要
    #從問題中提取產品類型（在RAGRetriever中已實現）
    retriever = RAGRetriever()
    
    #檢索相關商品評論
    matched_reviews = retriever.retrieve(
        query=question,
        max_products=3,  #最多回傳3個產品
        max_reviews_per_product=50  #每個產品最多50條評論
    )
    
    if not matched_reviews:
        return None, None
        
    #對評論進行情感分析
    labeled_reviews = analyze_sentiments(matched_reviews)
    
    #按產品和情感分組評論
    product_reviews_dict, stats_summary = group_reviews_by_product_and_sentiment(labeled_reviews)
    
    #生成每個產品各情感類別的評論摘要
    product_summaries = summarize_reviews_by_product(product_reviews_dict)

    return stats_summary, product_summaries

def answer_user_question(question: str):
    #回答用戶問題的主函數
    print(f"用戶提問：{question}")
    stats_summary, summaries = summarize_reviews(question)
    
    if not stats_summary:
        return "未找到相關商品評論。"

    result = "商品評論分析結果：\n\n"
    
    #添加每個商品的統計數據和總結
    for product, stats in stats_summary.items():
        total = sum(stats.values())
        result += f"🔹 {product}：\n"
        result += f"  評論統計：正評 {stats['正評']}、中立 {stats['中立']}、負評 {stats['負評']}（共 {total} 條）\n\n"
        
        #添加情感總結
        if product in summaries:
            for label in ["正評", "中立", "負評"]:
                if label in summaries[product] and stats[label] > 0:
                    result += f"  {label}觀點：\n{summaries[product][label]}\n\n"
    
    return result

#測試用
if __name__ == "__main__":
    question = "Which TV is better?"
    print(answer_user_question(question))