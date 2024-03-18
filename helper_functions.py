import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def drop_readings(dataframe: pd.DataFrame, emotion: str) -> pd.DataFrame:
    """
    Drops readings with a specified emotion from the provided DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        emotion (str): The emotion to drop readings for.

    Returns:
        pd.DataFrame: The DataFrame with readings corresponding to the specified emotion dropped.
    """
    dataframe.drop(dataframe[dataframe['label'] == emotion].index, inplace=True)
