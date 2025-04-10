# RAG-Product-Review-Insights
## Overview

This project analyzes product reviews using sentiment analysis and Retrieval-Augmented Generation (RAG). It helps users get insights into the sentiment distribution and key points from product reviews.

## Dataset

The project uses the [**Amazon Reviews 2023**](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset from Hugging Face.

## Technologies Used

### ETL (Extract, Transform, Load)
- **Data Extraction**: The project extracts data from the **Amazon Reviews 2023** dataset, which includes user reviews and product metadata.
- **Data Transformation**: The reviews are cleaned, filtered (e.g., removing short reviews), and transformed into a structured format for sentiment analysis and RAG.
- **Data Loading**: Transformed data is saved in CSV format and tokenized for machine learning and RAG processes.

### RAG (Retrieval-Augmented Generation)
- **Retrieval**: RAG is used to retrieve the most relevant product reviews based on a user's query (e.g., "Which TV is better?").
- **Augmentation**: RAG augments the retrieval process with additional context, providing summarized insights from top reviews for each sentiment category (positive, neutral, and negative).

### Machine Learning (Sentiment Analysis)
- **Sentiment Classification Model**: A **BERT-based** model is trained to classify reviews into **Positive**, **Neutral**, or **Negative** sentiments. This model is fine-tuned using the Amazon Reviews dataset and used to analyze the sentiment of each review.
- **Training**: The model is trained on the preprocessed review dataset and evaluated for accuracy, F1 score, and other metrics.

### AI (GPT-3 for Summarization)
- **Summarization**: GPT-3 is used to generate summaries of the top reviews for each product. These summaries focus on the main points mentioned in the positive, neutral, and negative reviews.
- **Query Handling**: GPT-3 also helps to refine responses based on user queries, generating meaningful insights from the retrieved reviews.
