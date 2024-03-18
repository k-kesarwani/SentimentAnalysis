"""
Description: This module contains the model architecture and related functions.
Date: 08 Mar 2024
Author: Kushagra Kesarwani
"""

import tensorflow as tf
from keras import layers, Sequential

class SentimentModel_v0(tf.keras.Model):
    """
    A sentiment analysis model architecture composed of text vectorization, LSTM layers, and dense output layers.
    """
    def __init__(self, vocab_size: int, output_features: int, units: int, dropout: float):
        super(SentimentModel, self).__init__()
        
        self.text_vect_layer = layers.TextVectorization(max_tokens=1500)   # Needs to be defined in main.py
        
        self.input_layer = Sequential([
            self.text_vect_layer,
            layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True)
        ])
        
        self.lstm_layer = Sequential([
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),

            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),

            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),

            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
        ])
        
        self.output_layer = Sequential([
            layers.LSTM(units),
            layers.Dropout(dropout),
            layers.Dense(units, activation='relu'),
            layers.Dense(output_features, activation='sigmoid')
        ])

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.lstm_layer(x)
        x = self.output_layer(x)
        return x

class SentimentModel_v1(tf.keras.Model):
    """
    A sentiment analysis model architecture composed of text vectorization, GRU layers, and dense output layers.
    """
    def __init__(self, vocab_size: int, 
                 output_features: int, 
                 units: int, 
                 dropout: float):
        super(SentimentModel, self).__init__()
        
        self.text_vect_layer = layers.TextVectorization(max_tokens=1500) # Needs to be defined in main.py
        
        self.input_layer = Sequential([
            self.text_vect_layer,
            layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True)
        ])
        
        self.gru_layer = Sequential([
            layers.GRU(units, return_sequences=True),
            layers.Dropout(dropout),

            layers.GRU(units, return_sequences=True),
            layers.Dropout(dropout),

            layers.GRU(units, return_sequences=True),
            layers.Dropout(dropout),

            layers.GRU(units, return_sequences=True),
            layers.Dropout(dropout),
        ])
        
        self.output_layer = Sequential([
            layers.LSTM(units),
            layers.Dropout(dropout),
            layers.Dense(units, activation='relu'),
            layers.Dense(output_features, activation='sigmoid')
        ])

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.lstm_layer(x)
        x = self.output_layer(x)
        return x

def train_model(model: tf.keras.Model, 
                train_set: tf.data.Dataset, 
                val_set: tf.data.Dataset, 
                epochs: int=10):

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_set, validation_data=val_set, epochs=epochs)
    return history

def save_model_weights(model: tf.keras.Model, 
                       weights_path: str):
    model.save_weights(weights_path)
    print("Model weights saved successfully.")

def evaluate_model(model: tf.keras.Model, 
                   test_set: tf.data.Dataset):
                   
    test_loss, test_accuracy = model.evaluate(test_set, verbose=False)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)

def predict_emotion(model, text):
    predictions = model.predict(text)
    return predictions
