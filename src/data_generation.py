import pandas as pd
import numpy as np

def generate_supply_chain_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'order_id': range(1, n_samples + 1),
        'product_id': np.random.randint(1, 101, n_samples),
        'quantity': np.random.randint(1, 101, n_samples),
        'supplier_id': np.random.randint(1, 11, n_samples),
        'order_date': pd.date_range(start='2023-01-01', periods=n_samples),
        'delivery_date': None,
        'transportation_mode': np.random.choice(['Air', 'Sea', 'Road'], n_samples),
        'distance': np.random.uniform(50, 1000, n_samples),
        'weather_condition': np.random.choice(['Good', 'Bad', 'Extreme'], n_samples),
        'delay': None
    }

    df = pd.DataFrame(data)
    
    # Calculate delivery date and delay
    df['delivery_date'] = df['order_date'] + pd.Timedelta(days=5)  # Base delivery time
    df.loc[df['transportation_mode'] == 'Sea', 'delivery_date'] += pd.Timedelta(days=7)
    df.loc[df['weather_condition'] == 'Bad', 'delivery_date'] += pd.Timedelta(days=2)
    df.loc[df['weather_condition'] == 'Extreme', 'delivery_date'] += pd.Timedelta(days=5)
    
    df['delay'] = (df['delivery_date'] - df['order_date']).dt.days - 5  # Delay in days
    
    return df

if __name__ == "__main__":
    df = generate_supply_chain_data()
    df.to_csv('data/synthetic_data.csv', index=False)
    print("Synthetic data generated and saved to 'data/synthetic_data.csv'")