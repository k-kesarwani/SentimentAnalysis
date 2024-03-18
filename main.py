"""
Description: This module contains the main code for data loading, training, evaluation, and inference.
Date: 08 March 2024
Author: Kushagra Kesarwani
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import helper_functions
import models

DIR = 'kaggle/input/emotions-dataset-for-nlp/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
TEST_FILE = 'test.txt'

train_df = pd.read_csv(DIR + TRAIN_FILE, delimiter=';', header=None, names=['sentence', 'label'])
val_df = pd.read_csv(DIR + VAL_FILE, delimiter=';', header=None, names=['sentence', 'label'])
test_df = pd.read_csv(DIR + TEST_FILE, delimiter=';', header=None, names=['sentence', 'label'])

# Preprocessing
train_df['processed_text'] = train_df['sentence'].apply(helper_functions.preprocess_text)
val_df['processed_text'] = val_df['sentence'].apply(helper_functions.preprocess_text)
test_df['processed_text'] = test_df['sentence'].apply(helper_functions.preprocess_text)

encoder = LabelEncoder()
train_df['label_num'] = encoder.fit_transform(train_df['label'])
val_df['label_num'] = encoder.transform(val_df['label'])
test_df['label_num'] = encoder.transform(test_df['label'])

# Model training
model = models.SentimentModel(vocab_size=1500, output_features=len(encoder.classes_), UNITS=64, DROPOUT=0.2)
train_set = tf.data.Dataset.from_tensor_slices((train_df['processed_text'].values, train_df['label_num'].values)).batch(32)
val_set = tf.data.Dataset.from_tensor_slices((val_df['processed_text'].values, val_df['label_num'].values)).batch(32)
history = models.train_model(model, train_set, val_set, epochs=10)

# Evaluation
test_set = tf.data.Dataset.from_tensor_slices((test_df['processed_text'].values, test_df['label_num'].values)).batch(32)
models.evaluate_model(model, test_set)

# Save model weights
weights_path = 'sentiment_model_weights.h5'
models.save_model_weights(model, weights_path)

# Inference
sample_text = test_df['processed_text'].values
predictions = models.predict_emotion(model, sample_text)
print(predictions[0])
