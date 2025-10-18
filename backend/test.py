import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load data and models
df = pd.read_csv('phishing_site_urls.csv', encoding='latin1')
hybrid_model = load_model('hybrid_model_new.keras')
ensemble_model = joblib.load('random_forest_model.pkl')

# Tokenizer and tfidf
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['URL'])
tfidf_vectorizer = TfidfVectorizer(max_features=1)
tfidf_vectorizer.fit(df['URL'])

# Input preprocessing
print("Enter a URL to check:")
url = input()
url_seq = tokenizer.texts_to_sequences([url])
url_pad = pad_sequences(url_seq, maxlen=177)
url_tfidf = tfidf_vectorizer.transform([url])

# Deep learning prediction
dl_pred = hybrid_model.predict(url_pad)
dl_result = "Phishing" if dl_pred > 0.5 else "Legitimate"

# Ensemble prediction
ensemble_pred = ensemble_model.predict(url_tfidf)
ensemble_result = "Phishing" if ensemble_pred[0] == 1 else "Legitimate"

# Combine predictions
combined_prediction = (dl_pred + ensemble_pred) / 2
combined_result = "Phishing" if combined_prediction > 0.5 else "Legitimate"

# Print results
print("Deep Learning Prediction:", dl_result)
print("Ensemble Prediction:", ensemble_result)
print("Combined Prediction:", combined_result)
