from faker import Faker
import numpy as np
import pandas as pd


def generate_merchants(n=1000):
    fake = Faker('en_GB')
    uk_categories = ["Supermarket", "Clothing", "Electronics", "Restaurants", "Travel", "Finance", "Pharmacy", "Books",
                     "Sports", "Health & Beauty"]

    merchants = [{
        'merchant_id': fake.uuid4(),
        'name': fake.company(),
        'category': np.random.choice(uk_categories),
        'description': fake.catch_phrase(),
        'location': fake.city()
    } for _ in range(n)]

    return pd.DataFrame(merchants)


def generate_customers(merchants_df, n=10000):
    fake = Faker('en_GB')
    customers = []
    for _ in range(n):
        customers.append({
            'customer_id': fake.uuid4(),
            'age': np.random.randint(18, 75),
            'income': np.random.randint(15000, 100000),
            'location': fake.city(),
            'transactions': merchants_df.sample(5)['merchant_id'].values.tolist()
        })
    return pd.DataFrame(customers)
