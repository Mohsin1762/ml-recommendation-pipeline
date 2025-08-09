from fastapi import FastAPI, HTTPException
import joblib
from app.recommend import get_recommendations


app = FastAPI()

# Load data once on startup
merchants_df = joblib.load('models/merchants.pkl')
customers_df = joblib.load('models/customers.pkl')

@app.get("/recommend/{index}")
def recommend(index: int, top_k: int = 10):
    if index < 0 or index >= len(customers_df):
        raise HTTPException(status_code=404, detail="Invalid index")

    customer_id = customers_df.iloc[index]['customer_id']
    recommendations = get_recommendations(customer_id, merchants_df, customers_df, top_k)
    if recommendations.empty:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return recommendations.to_dict(orient='records')


@app.get("/customers")
def list_customers():
    return customers_df['customer_id'].tolist()
