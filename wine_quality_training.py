import os
import pickle
import time
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#load and split data
df_pq = pd.read_parquet("data/winequality.parquet")
y = df_pq["quality"]
X = df_pq.drop(columns=["quality", "wine_color"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#model selection and training:
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#model predictions:
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report: {report}")


#save model artifact
os.makedirs("models", exist_ok=True)
model_path = "models/wine_quality_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f" Saved model to: {model_path}")

#save evaluation metrics
metadata_path = "models/wine_quality_model.metadata.json"
metadata = {
    "model_type": "RandomForestClassifier",
    "accuracy": accuracy,
    "metrics_per_class": report
}

with open(metadata_path, "w")as f:
    json.dump(metadata, f, indent=4)
print(f"Saved model metadata to {metadata_path}")