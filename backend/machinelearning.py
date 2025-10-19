import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Load the phishing dataset
df = pd.read_csv("phishing_site_urls.csv", encoding='latin1')
# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('URL', axis=1), df['Label'], test_size=0.2, random_state=42)
# Create an ensemble model
model = VotingClassifier(estimators=[('rfc', RandomForestClassifier()), ('ada', AdaBoostClassifier()), ('gbc', GradientBoostingClassifier())], voting='hard')
# Train the model
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)   
print(classification_report(y_test, y_pred))
# Save the trained model

joblib.dump(model,'phishing_model.pkl')
