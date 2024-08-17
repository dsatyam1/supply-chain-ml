import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Convert date columns to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    
    # Create features
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['transportation_mode', 'weather_condition'])
    
    # Select features and target
    features = ['quantity', 'distance', 'month', 'day_of_week'] + \
               [col for col in df.columns if col.startswith(('transportation_mode_', 'weather_condition_'))]
    target = 'delay'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_data('data/synthetic_data.csv')
    print("Data preprocessed and split into train and test sets")