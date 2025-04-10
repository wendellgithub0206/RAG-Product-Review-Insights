import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class RAGRetriever:
    def __init__(
        self,
        index_path="./rag/faiss_index.index",
        source_text_path="./rag/source_texts.pkl",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k=100  #檢索更多結果，之後進行過濾
    ):
        print("載入向量模型與索引")
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(source_text_path, "rb") as f:
            self.source_texts = pickle.load(f)
        self.top_k = top_k

    def retrieve(self, query: str, max_products=3, max_reviews_per_product=50):
        #根據輸入問題檢索相關商品評論
        print(f"查詢問題: {query}")
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype("float32"), self.top_k)

        #收集產品標題和對應的評論文本
        product_reviews = defaultdict(list)
        product_categories = {}
        
        #檢測查詢中的產品類型
        query_lower = query.lower()
        product_type = None
        for keyword, type_name in [
            (["tv", "television"], "TV"),
            (["tablet", "ipad"], "Tablet"),
            (["phone", "smartphone"], "Phone"),
            (["laptop", "computer"], "Computer"),
            (["headphone", "headset", "earphone"], "Headphone")
        ]:
            if any(k in query_lower for k in keyword):
                product_type = type_name
                break
        
        #遍歷所有檢索結果
        for i in I[0]:
            product_title = self.source_texts[i]['product_title']
            category = self.source_texts[i]['category']
            review_text = self.source_texts[i]['review_text']
            
            #如果指定了產品類型，只保留相關產品
            if product_type:
                product_title_lower = product_title.lower()
                category_lower = category.lower()
                
                if product_type == "TV" and any(k in product_title_lower or k in category_lower for k in ["tv", "television"]):
                    product_reviews[product_title].append(review_text)
                    product_categories[product_title] = category
                elif product_type == "Tablet" and any(k in product_title_lower or k in category_lower for k in ["tablet", "ipad"]):
                    product_reviews[product_title].append(review_text)
                    product_categories[product_title] = category
                elif product_type == "Phone" and any(k in product_title_lower or k in category_lower for k in ["phone", "smartphone"]):
                    product_reviews[product_title].append(review_text)
                    product_categories[product_title] = category
                elif product_type == "Computer" and any(k in product_title_lower or k in category_lower for k in ["laptop", "computer"]):
                    product_reviews[product_title].append(review_text)
                    product_categories[product_title] = category
                elif product_type == "Headphone" and any(k in product_title_lower or k in category_lower for k in ["headphone", "headset", "earphone"]):
                    product_reviews[product_title].append(review_text)
                    product_categories[product_title] = category
            else:
                #沒有指定產品類型，保留所有檢索結果
                product_reviews[product_title].append(review_text)
                product_categories[product_title] = category
        
        #如果没有找到符合產品類型的商品，返回前max_products個最相關商品
        if not product_reviews and product_type:
            print(f"未找到{product_type}類產品，返回最相關的商品")
            products_added = set()
            for i in I[0]:
                product_title = self.source_texts[i]['product_title']
                if product_title not in products_added and len(products_added) < max_products:
                    product_reviews[product_title].append(self.source_texts[i]['review_text'])
                    product_categories[product_title] = self.source_texts[i]['category']
                    products_added.add(product_title)
                elif product_title in product_reviews:
                    product_reviews[product_title].append(self.source_texts[i]['review_text'])
        
        #選擇評論數量最多的前max_products個商品
        most_reviewed_products = sorted(product_reviews.keys(), 
                                        key=lambda x: len(product_reviews[x]), 
                                        reverse=True)[:max_products]
        
        #整理結果
        results = []
        for product in most_reviewed_products:
            #對每個商品最多取max_reviews_per_product條評論
            reviews = product_reviews[product][:max_reviews_per_product]
            results.append({
                "product": product,
                "category": product_categories[product],
                "reviews": reviews  #保持為列表形式
            })
        
        return results