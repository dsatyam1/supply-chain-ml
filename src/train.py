from data_preprocessing import preprocess_data
from model import train_model, evaluate_model, save_model

def main():
    X_train, X_test, y_train, y_test, scaler = preprocess_data('data/synthetic_data.csv')
    
    model = train_model(X_train, y_train)
    
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Model performance: MSE = {mse:.2f}, R2 = {r2:.2f}")
    
    save_model(model, 'model.joblib')
    print("Model saved as 'model.joblib'")

if __name__ == "__main__":
    main()