ML Recommendation System Pipeline:
Project Overview

This project is a complete end-to-end recommendation system pipeline built using Python, FastAPI, and Docker.
It leverages customer and merchant data to generate personalized merchant recommendations based using BERT (Transformers) on similarity embeddings.

The system exposes a scalable API for serving recommendations and is containerized using Docker for easy deployment and sharing.


Customer-Merchant Embedding Generation:
Merchant and customer profiles are embedded using vector representations to enable similarity-based recommendations.
Personalized Merchant Recommendations:
Given a customer ID, the API returns the top K recommended merchants using cosine similarity of embeddings.
FastAPI REST API:
A fast, asynchronous API exposes endpoints to get recommendations and list customers.
Dockerized Deployment:
The entire system is containerized for easy deployment, scaling, and sharing.
Data Persistence:
Preprocessed data stored as pickle files for efficient loading and serving.


Customer & Merchant Data
          │
          ▼
Embedding Generation (using embedding function)
          │
          ▼
Similarity Calculation (Cosine similarity)
          │
          ▼
Recommendation Engine (Top-K retrieval)
          │
          ▼
FastAPI REST API
          │
          ▼
Docker Container
          │
          ▼
Accessible Recommendation Service



Tech Stack:
    Python 3.8+
    FastAPI
    scikit-learn (for cosine similarity)
    NumPy & pandas
    Transformers
    Faker ( for data creation )
    joblib (for loading preprocessed data)
    Docker (for containerization)
