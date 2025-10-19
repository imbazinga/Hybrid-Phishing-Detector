# FishBot
### Create an intelligent system using AI/ML to detect phishing domains which imitate look and feel of genuine domains (SIH 2023 Problem Statement - 1454).

An AI/ML-Based System for Detecting Phishing Domains

## Overview
**FishBot** is a cybersecurity project created during the **Smart India Hackathon (SIH)**. The system automatically detects **phishing websites** that mimic the appearance of legitimate domains. It uses a mix of **Machine Learning (ML)** and **Deep Learning (DL)** models, along with **SSL certificate validation**.

The project shows how AI can protect users from online scams by examining URL structures, patterns, and web security features.

## Objectives
- To automatically detect **phishing URLs** using machine learning and deep learning.
- To analyze a given website’s **SSL certificate validity**.
- To provide a **simple and interactive web interface** for testing URLs.
- To demonstrate a **hybrid detection approach** that combines AI model predictions and SSL verification.

## Models Used
This system follows a hybrid detection approach combining multiple models and verification checks:
1. **Deep Learning Model (LSTM):**
- Analyzes the structure and sequence of characters in URLs.
- Learns deep representations to differentiate phishing patterns from legitimate domains.

2. **Machine Learning Model (Ensemble):**
- Combines multiple ML algorithms such as Random Forest, AdaBoost, and Gradient Boosting.
- Uses extracted URL-based features and applies a voting mechanism for stable predictions.

3. **SSL Certificate Validation:**
- Verifies if the target domain has a valid HTTPS certificate.
- Acts as an additional factor in determining overall trustworthiness.

## Hybrid Logic
```bash
final_score = (DL_score + ML_prediction) / 2
if final_score > 0.5 or SSL_invalid:
    verdict = “Phishing”
else:
    verdict = “Legitimate”
```
## Dataset
The dataset used for training consisted of both phishing and legitimate URLs, collected from publicly available sources such as Kaggle phishing detection datasets,etc.

Each entry was labeled:
```bash
1 → Phishing
0 → Legitimate
```

## Results
While the exact metrics from the original experiment are unavailable, the system produced consistent and accurate classifications during testing.

Example Output:
```bash
URL: free-gift-login.com
Protocol: https
Domain: free-gift-login.com
SSL Certificate: Invalid 
Result: Phishing 
```

## Project Structure
```plaintext
hybrid-phishing-detector/
├── backend/
│ ├── backend.py # Flask API for SSL & URL info checker
│ ├── phishing_detection.py # Hybrid phishing detection logic
│ ├── deeplearning.py # Deep learning (LSTM) model training
│ ├── machinelearning.py # Ensemble ML model training
│ ├── urlscanner.py # Basic SSL validator
│ ├── test.py # Test script for predictions
│ └── TrainingScript.py # Combined training script
│
├── frontend/
│ └── frontend.html # Simple HTML interface for testing
│
├── models/
│ ├── ensemble_model.pkl
│ └── hybrid_model_new.keras 
│
├── data/
│ └── phishing_site_urls.csv 
│
├── requirements.txt
└── README.md
```

## Setup & Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/imbazinga/Hybrid-Phishing-Detector
cd Hybrid-phishing-detector
```
Due to GitHub size limits, the trained models are available here:
([link](https://drive.google.com/drive/folders/1xbJlBKJ1fNGMUBKkZDDYE-2hwVOP6mlj?usp=drive_link))

### Step 2: Install the Dependencies
```bash
pip install -r requirements.txt
```
Before training or running the system, make sure the file **phishing_site_urls.csv** is placed in the same folder as the training script or inside the project root directory. The script will automatically load it from there.

### Step 3: Run the Backend
```bash
cd backend
python backend.py
```
And you should see a message like this:
Server running at http://localhost:5000

### Step 4: Open the Frontend
Open **frontend/frontend.html** and enter any URL to scan (for example: google.com).

## Technologies Used
- AI/ML Frameworks: TensorFlow, Keras, Scikit-learn
- Programming Language: Python
- Backend Framework: Flask
- Frontend: HTML, CSS, JavaScript
- Libraries: NumPy, Pandas, Requests, Joblib

## Contributions
Contributions to the FishBot project are welcome! If you have any ideas for improvements, bug fixes, or new features, feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See `LICENSE` file for more details.
