from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_model import train_xgboost
from src.evaluate_model import evaluate_model

DATA_PATH = "data/customer_churn_120k.csv"

def main():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_xgboost(X_train, y_train)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
