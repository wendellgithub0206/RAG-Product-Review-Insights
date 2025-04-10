# RAG-Product-Review-Insights
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


