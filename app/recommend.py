import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_merchant_embeddings(merchants_df, embedding_func):
    merchants_df['embedding'] = merchants_df['description'].apply(embedding_func)
    return merchants_df

def create_customer_embeddings(customers_df, merchants_df):
    # Example: simple dummy embeddings for now
    customers_df['embedding'] = [np.random.rand(768) for _ in range(len(customers_df))]
    return customers_df

def recommend_top_merchants(customer_id, merchants_df, customers_df, top_k=10):
    customer_row = customers_df[customers_df['customer_id'] == customer_id]
    if customer_row.empty:
        return []

    customer_emb = customer_row['embedding'].values[0]
    merchant_embs = np.vstack(merchants_df['embedding'].values)

    sims = cosine_similarity(customer_emb.reshape(1, -1), merchant_embs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    recommended_merchants = merchants_df.iloc[top_indices]
    return recommended_merchants[['merchant_id', 'name', 'category', 'description']]

def get_recommendations(customer_id, merchants_df, customers_df, top_k=10):
    return recommend_top_merchants(customer_id, merchants_df, customers_df, top_k)



def load_data():
    merchants_df = joblib.load('models/merchants.pkl')
    customers_df = joblib.load('models/customers.pkl')
    return merchants_df, customers_df
