import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("⏳ Loading data...")
df = pd.read_csv('cleaned_data.csv')
df['clean_text'] = df['clean_text'].fillna("")

# Separate features
X_text = df['clean_text']
X_numeric = df[['text_len', 'has_salary', 'profile_missing', 'telecommuting', 'has_company_logo', 'has_questions']]
y = df['fraudulent']

print("🧠 Training AI Model (This might take a minute)...")
# Vectorize text
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_text_tfidf = tfidf.fit_transform(X_text).toarray()

# Scale numeric data
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_numeric)

# Combine and Train
X_final = np.hstack((X_text_tfidf, X_num_scaled))
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_final, y)

print("💾 Saving Model Files...")
# Export the components
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'numeric_scaler.pkl')
joblib.dump(model, 'fake_job_model.pkl')

print("✅ Success! Your .pkl files are ready.")