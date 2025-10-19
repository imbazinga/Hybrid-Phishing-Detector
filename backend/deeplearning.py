import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# Load your preprocessed dataset
data = pd.read_csv("phishing_site_urls.csv", encoding='latin1')

# Drop rows with NaN values in the target variable 'Label'
data = data.dropna(subset=['Label'])

# Preprocessing for deep learning model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['URL'])
X_text = data['URL']
X_sequences = tokenizer.texts_to_sequences(X_text)
X = pad_sequences(X_sequences)
y = data['Label']

# Split the data into training and testing sets for deep learning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weights for imbalanced data
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Define parameters for the deep learning model
input_dim = len(tokenizer.word_index) + 1
embedding_dim = 128
max_sequence_length = X.shape[1]

# Create a hybrid model combining LSTM and MLP
hybrid_model = Sequential()
hybrid_model.add(Embedding(
    input_dim=input_dim,
    output_dim=embedding_dim,
    input_length=max_sequence_length))
hybrid_model.add(LSTM(128, return_sequences=True))
hybrid_model.add(LSTM(128))
hybrid_model.add(Dense(64, activation='relu'))
hybrid_model.add(Dropout(0.5))
hybrid_model.add(Dense(1, activation='sigmoid'))

# Implement a learning rate scheduler
initial_learning_rate = 0.001

def lr_scheduler(epoch):
    return initial_learning_rate * 0.9 ** epoch

opt = Adam(learning_rate=initial_learning_rate)
hybrid_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
lr_callback = LearningRateScheduler(lr_scheduler)

# Implement early stopping for deep learning
early_stopping_dl = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the hybrid model
print("Training the hybrid model (LSTM + MLP)...")
hybrid_model.fit(X_train, y_train, epochs=5, batch_size=2000, validation_data=(X_test, y_test),
                 class_weight={0: class_weights[0], 1: class_weights[1]}, callbacks=[lr_callback, early_stopping_dl], verbose=1)

# Evaluate the deep learning model on test data
y_pred_hybrid = (hybrid_model.predict(X_test) > 0.5).astype("int32")
accuracy_hybrid = accuracy_score(y_test, y_pred_hybrid)
print(f"Deep Learning Model Accuracy: {accuracy_hybrid}")
print(classification_report(y_test, y_pred_hybrid))

# Save the trained model
hybrid_model.save('hybrid_model_new.keras')

