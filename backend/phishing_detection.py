import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import requests
from requests.exceptions import RequestException, ConnectTimeout

# Function to check SSL certificate
def check_ssl_certificate(url):
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url  # Prepend "https://" if not provided in the URL

    try:
        response = requests.head(url, timeout=10)  # Set a timeout for the request
        if response.status_code == 200:
            if response.headers.get("Server"):
                print(f"The website {url} has a valid SSL certificate.")
            else:
                print(f"The website {url} does not have a valid SSL certificate. It may be a phishing site.")
                return False  # Return False if SSL certificate is not valid
        else:
            print(f"Failed to connect to {url}. Status code: {response.status_code}")
    except ConnectTimeout:
        print(f"Connection to {url} timed out. Please check the URL or your internet connection.")
    except RequestException as e:
        print(f"An error occurred while checking {url}: {str(e)}")

    return True  # Return True if SSL certificate is valid or an error occurred

# Load data and models
df = pd.read_csv('phishing_site_urls.csv', encoding='latin1')
hybrid_model = load_model('hybrid_model_new.keras')
ensemble_model = joblib.load('ensemble_model.pkl')

# Tokenizer and tfidf
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['URL'])
tfidf_vectorizer = TfidfVectorizer(max_features=277)
tfidf_vectorizer.fit(df['URL'])

# Input preprocessing
print("Enter a URL to check:")
url = input()

# Check SSL certificate
ssl_valid = check_ssl_certificate(url)

if not ssl_valid:
    # If SSL certificate is not valid, consider it a phishing site
    print("Phishing Site (SSL certificate not valid)")
else:
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
    print("PHISHING Prediction:", dl_result)
    #print("Ensemble Prediction:", ensemble_result)
    #print("Combined Prediction:", combined_result)
