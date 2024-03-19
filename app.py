import gradio as gr
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import spacy
from timeit import default_timer as timer

with open('class_names.txt', 'r') as f:
    labels = [emotion.strip() for emotion in f.readlines()] 

with open('examples.txt', 'r') as f:
    example_list = [example.strip() for example in f.readlines()]

encoder= LabelEncoder()
encoder.fit(labels) 

nlp = spacy.load("en_core_web_sm")

model = tf.keras.models.load_model('models/gru_model.keras')

def preprocess_single_sentence(sentence):
    """
    Preprocesses a single sentence.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Preprocessed and tokenized sentence.
    """
    processed_text = ' '.join([token.lemma_ for token in nlp(sentence) if not token.is_stop and not token.is_punct and not token.is_space])
    return processed_text

def make_predictions(text):
    """
    Make predictions on the given text using the trained model.

    Args:
        text (str): The text to make predictions on.

    Returns:
        list: A list of predictions.
    """
    text= preprocess_single_sentence(text)
    text= tf.expand_dims(text, 0)

    start_time= timer()
    probability = model.predict(text)
    pred_label_with_prob= {labels[i]: float(probability[0][i]) for i in range(len(labels))} 
    pred_time = round(timer() - start_time, 5)
    return pred_label_with_prob, pred_time

input= gr.Textbox(lines=5, label="Enter text", placeholder="i like to have the same breathless feeling as a reader eager to see what will happen next")
outputs=[
        gr.Label(num_top_classes=len(labels), label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ]
title= ' Sentiment Analysis 🤣😱😡😢 '
description= 'The sentiment analysis model is a deep learning-based natural language processing (NLP) model designed to analyze and classify the sentiment expressed in text data. It is trained to understand the emotional tone of text and categorize it into predefined sentiment categories such as <b>anger, fear, saddness and joy.<b>'
demo= gr.Interface(fn=make_predictions, 
                   inputs=input, 
                   outputs=outputs, 
                   title=title, 
                   description=description,
                   examples= example_list
                  )
demo.launch()