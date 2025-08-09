from data_generator import generate_merchants, generate_customers
from bert_embedding import get_bert_embedding
from recommend import create_merchant_embeddings, create_customer_embeddings, recommend_top_merchants
import pandas as pd
import numpy as np


def main():
    merchants_df = generate_merchants()
    customers_df = generate_customers(merchants_df)

    merchants_df = create_merchant_embeddings(merchants_df, get_bert_embedding)
    customers_df = create_customer_embeddings(customers_df, merchants_df)

    merchant_embeddings = np.vstack(merchants_df['embedding'].values)

    # Example: Recommend for first 3 customers
    for i in range(3):
        top_indices = recommend_top_merchants(customers_df['embedding'][i], merchant_embeddings, top_k=5)
        print(f"\nTop 5 merchants recommended for Customer {i}:")
        print(merchants_df.iloc[top_indices][['name', 'category']])


if __name__ == "__main__":
    main()
