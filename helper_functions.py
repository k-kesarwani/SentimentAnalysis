import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import spacy

with open('kaggle/label_names.txt', 'r') as f:
    labels = [emotion.strip() for emotion in f.readlines()] 
    
encoder= LabelEncoder()
encoder.fit(labels)

nlp = spacy.load("en_core_web_sm")


def plot_pie_chart(data_frame: pd.DataFrame, title: str) -> None:
    """
    Plot a pie chart to visualize label distribution in the provided DataFrame.

    Args:
        data_frame (pd.DataFrame): The DataFrame containing the data to visualize.
        title (str): The title for the pie chart.

    Returns:
        None
    """
    label_count = data_frame['label'].value_counts()
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    plt.pie(label_count, labels=label_count.index, colors=sns.color_palette("hls", len(label_count.index)), autopct='%1.1f%%', startangle=90)
    plt.title(f"{title} Label Distribution")
    plt.show()
    plt.close()

def preprocess_text(df: pd.DataFrame, emotions: list=['love', 'surprise']):
    """
    Preprocesses text data in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'sentence' and 'label' columns.
        encoder (LabelEncoder): Label encoder for the labels.
        emotions (list): List of emotions to drop from the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with preprocessed text and encoded labels.
    # """
    for i in emotions:
        df = df[df['label'] != i]

    df['processed_text'] = df['text'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_stop and not token.is_punct and not token.is_space]))

    df['label_num'] = encoder.transform(df['label'])
    df.drop(columns=['text', 'label'], inplace=True)
    return df

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

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss curves.

    Parameters:
    history (History object): History object returned by model.fit() containing training metrics.

    Returns:
    None
    """
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
