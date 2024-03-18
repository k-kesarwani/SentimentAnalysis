# Emotion Recognition Model

This repository contains code for building and evaluating an emotion recognition model using TensorFlow. The model is trained on a dataset from Kaggle containing text samples labeled with different emotions.

## Dataset
The dataset used for training, validation, and testing is located in the directory `kaggle/input/emotion-dataset-for-nlp/`. It consists of three files:
- `train.txt`: Training data
- `val.txt`: Validation data
- `test.txt`: Testing data

The data is loaded into Pandas DataFrames and preprocessed before training the model.

## Preprocessing
Text data preprocessing involves the following steps:
1. Removing specified emotions from the dataset.
2. Tokenization and lemmatization using spaCy.
3. Encoding labels using sklearn's LabelEncoder.

## Model Architecture
The model architecture consists of the following layers:
1. TextVectorization layer for tokenizing text.
2. Embedding layer for word embeddings.
3. Two GRU (Gated Recurrent Unit) layers for sequence processing.
4. Dense output layer with softmax activation for multi-class classification.

## Training
The model is trained using TensorFlow's `Sequential` API. Training is conducted with a specified batch size and number of epochs.

## Evaluation
The model is evaluated on both the validation and test sets using accuracy as the metric. Additionally, the model is saved for future use.

## Making Predictions
To make predictions on a single sentence, the following steps are performed:
1. Preprocess the sentence to remove extra tokens.
2. Expand the dimension to match the model's input shape.
3. Pass the preprocessed sentence through the model for prediction.

## Usage
To train the model and evaluate its performance, run the provided Python script. You can also load the saved model to make predictions on new data.

```python
python emotion_recognition_model.py
