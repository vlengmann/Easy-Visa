## Final Model Code

print("train.py STARTED - top of file reached")

def main():
    print("main() is running - training begins")

    # ---------------------------------------------------------
    # Imports
    # ---------------------------------------------------------
    import os
    import numpy as np
    import joblib
    import json
    import pandas as pd
    from sklearn import metrics
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import confusion_matrix
    from xgboost import XGBClassifier

    from utils.data_ingestion import DataIngestorFactory
    from model.preprocess import preprocess_data   # ✅ Correct import

    # ---------------------------------------------------------
    # Ensure model folder exists
    # ---------------------------------------------------------
    os.makedirs("model", exist_ok=True)

    # ---------------------------------------------------------
    # 1. Ingest Data
    # ---------------------------------------------------------
    print("Step 1: Ingesting data...")
    file_path = "data/visa_data.csv"
    file_extension = os.path.splitext(file_path)[1]

    ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = ingestor.ingest(file_path)
    print("Data ingestion complete.")

    # ---------------------------------------------------------
    # 2. Preprocess Data
    # ---------------------------------------------------------
    print("Step 2: Preprocessing data...")
    processed = preprocess_data(df, verbose=False)
    X = processed["X"]
    y = processed["y"]
    feature_columns = processed["feature_columns"]
    print("Preprocessing complete.")

    # ---------------------------------------------------------
    # 3. Train/Test Split
    # ---------------------------------------------------------
    print("Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )
    print("Train/test split complete.")

    # ---------------------------------------------------------
    # 4. XGBoost Hyperparameter Tuning
    # ---------------------------------------------------------
    print("Step 4: Running GridSearchCV...")
    xgb_tuned = XGBClassifier(random_state=1, eval_metric='logloss')

    param_grid = {
        'n_estimators': np.arange(50, 110, 25),
        'scale_pos_weight': [1, 2, 5],
        'learning_rate': [0.01, 0.1, 0.05],
        'gamma': [1, 3],
        'subsample': [0.7, 0.9]
    }

    scorer = metrics.make_scorer(metrics.f1_score)

    grid_obj = GridSearchCV(
        xgb_tuned,
        param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1
    )

    grid_obj = grid_obj.fit(X_train, y_train)
    best_model = grid_obj.best_estimator_
    print("GridSearchCV complete.")

    # ---------------------------------------------------------
    # 5. Fit Final Model
    # ---------------------------------------------------------
    print("Step 5: Fitting final model...")
    best_model.fit(X_train, y_train)
    print("Model fit complete.")

    # ---------------------------------------------------------
    # 6. Compute Metrics
    # ---------------------------------------------------------
    print("Step 6: Computing metrics...")
    y_pred = best_model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

    with open("model/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print("[SAVED] metrics.json")

    # ---------------------------------------------------------
    # 7. Save Confusion Matrix
    # ---------------------------------------------------------
    print("Step 7: Saving confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    np.save("model/confusion_matrix.npy", cm)
    print("[SAVED] confusion_matrix.npy")

    # ---------------------------------------------------------
    # 8. Save Feature Importances
    # ---------------------------------------------------------
    print("Step 8: Saving feature importances...")
    np.save("model/feature_importances.npy", best_model.feature_importances_)
    print("[SAVED] feature_importances.npy")

    # ---------------------------------------------------------
    # 9. Save Model + Feature Columns
    # ---------------------------------------------------------
    print("Step 9: Saving model and feature columns...")
    joblib.dump(best_model, "model/model.pkl")
    print("[SAVED] model.pkl")

    joblib.dump(feature_columns, "model/feature_columns.pkl")
    print("[SAVED] feature_columns.pkl")

    print("Training complete. Artifacts saved.")


# ---------------------------------------------------------
# Run Script
# ---------------------------------------------------------
if __name__ == "__main__":
    main()