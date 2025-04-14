## Overview

This project analyzes product reviews using sentiment analysis and Retrieval-Augmented Generation (RAG). It helps users get insights into the sentiment distribution and key points from product reviews.

## Dataset

The project uses the [**Amazon Reviews 2023**](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset from Hugging Face.

## Technologies

### 1. ETL
- Extracts raw review data.
- Transforms reviews (removes short texts, filters necessary data, maps ratings to sentiment).
- Loads the data for tokenization and sentiment analysis.

### 2. Machine Learning (BERT)
- Trains a sentiment classification model on product reviews to predict sentiment (positive, neutral, negative).

### 3. RAG (Retrieval-Augmented Generation)
- Retrieves relevant products based on user queries.
- Performs sentiment analysis on the top reviews.
- Summarizes the sentiment distribution and top reviews for each product.

### 4. AI (GPT-3)
- Uses GPT-3 to generate summaries of the top reviews for each product, summarizing key points for positive, neutral, and negative sentiments.


### 5. EDA
![top_products](https://github.com/user-attachments/assets/7f4d9ad6-bcf0-4637-8cab-378e9a4e531e)
![sentiment_distribution](https://github.com/user-attachments/assets/253c7424-8df3-4cfb-8917-c521939a4e8f)
![review_length_distribution](https://github.com/user-attachments/assets/a92ba2bc-76a1-433e-8db0-bf8851a084e7)
![category_distribution](https://github.com/user-attachments/assets/3a707ff5-f766-47d5-9106-2075ad7acbd5)
