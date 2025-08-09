# Inside train_embeddings.py or any other .py file in app folder
from data_generator import generate_merchants, generate_customers
from bert_embedding import get_bert_embedding
from recommend import create_merchant_embeddings, create_customer_embeddings, recommend_top_merchants


import pandas as pd
import numpy as np
import joblib  # For saving models/data

def main():
    merchants_df = generate_merchants()
    customers_df = generate_customers(merchants_df)

    # Generate embeddings
    merchants_df = create_merchant_embeddings(merchants_df, get_bert_embedding)
    customers_df = create_customer_embeddings(customers_df, merchants_df)

    # Save the embeddings + data for API usage
    joblib.dump(merchants_df, 'models/merchants.pkl')
    joblib.dump(customers_df, 'models/customers.pkl')

    print("Saved merchant and customer embeddings.")

if __name__ == "__main__":
    main()
