from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    # Encode all categorical columns automatically
    categorical_cols = df.select_dtypes(include=["object"]).columns

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Target
    target_col = "churn"
    id_col = "customer_id"

    # Drop ID column and target column
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
