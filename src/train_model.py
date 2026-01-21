from xgboost import XGBClassifier
import joblib

def train_xgboost(X_train, y_train):
    # Calculate scale_pos_weight
    non_churn = (y_train == 0).sum()
    churn = (y_train == 1).sum()
    scale_pos_weight = non_churn / churn

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/xgboost_model.pkl")

    return model
