# Sentiment Analysis with GRU Model

This project aims to predict the emotion conveyed in a given text using a GRU (Gated Recurrent Unit) neural network model. The model is trained on a dataset containing text samples labeled with different emotions.

## Dataset
The dataset used for training, validation, and testing is the ["Emotion Dataset for NLP."](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) It consists of labeled text samples categorized into various emotions. The dataset is divided into training, validation, and test sets.

## Preprocessing
Text preprocessing involves cleaning the text data and tokenizing it using spaCy. Stop words, punctuation, and whitespace are removed from the text, and lemmatization is applied to obtain the base form of words. Emotions such as 'love' and 'surprise' are removed from the dataset during preprocessing.

## Model Architecture
The GRU model architecture consists of:
- Text vectorization layer
- Embedding layer
- Two GRU layers with dropout
- Dense output layer with softmax activation

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.

## Training
The model is trained using batches of text samples. Training is performed over multiple epochs, with validation performed on a separate validation set to monitor the model's performance.

## Evaluation
The trained model is evaluated on the test set to assess its accuracy in predicting emotions from unseen text samples.

## Making Predictions
To make predictions on new text samples, the following steps are followed:
1. Preprocess the text to remove unnecessary tokens.
2. Expand the dimensions of the preprocessed text.
3. Pass the preprocessed text through the trained model to obtain predictions.

## Demo Interface
A graphical interface (GUI) using Gradio is provided for making predictions on custom text inputs. Users can input text, and the model predicts the corresponding emotion conveyed in the text.

## File Structure
- **`SentimentAnalysis.ipynb`**: Jupyter Notebook containing the code for model training, evaluation, and prediction.
- **`gru_model.keras`**: Saved trained model file.
- **`helper_functions.py`**: Python script containing helper functions for data preprocessing and evaluation.
- **`README.md`**: This readme file providing an overview of the project.

## Requirements
- Python 3.x
- TensorFlow
- spaCy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- Gradio

## Usage
1. Clone the repository.
2. Install the required dependencies using:  
```python
pip install requirements.txt
```