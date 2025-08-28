import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import os

# Paths
DATA_PATH = "data/aes_dataset/training_set_rel3.tsv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
train_df = pd.read_csv(DATA_PATH, sep="\t", encoding="latin1")

print("✅ Data Loaded")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print(train_df.head(3))

# Extract essays and scores
X = train_df['essay']
y = train_df['domain1_score']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert essays to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_val_tfidf)

# Evaluate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"✅ Validation RMSE: {rmse:.4f}")

# Save model & vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "ridge_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
print("✅ Model and Vectorizer saved in models/")
