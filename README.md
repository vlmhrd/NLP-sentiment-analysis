# NLP-sentiment-analysis

# Goal
The primary goal of this project is to build and evaluate a machine learning model that can classify text data into multiple sentiment categories (positive, neutral, and negative) using natural language processing (NLP) techniques.

# Description
This project implements a multiclass sentiment analysis pipeline using Python and PyTorch. The workflow begins by loading a labeled dataset from Hugging Face, which contains text samples and their corresponding sentiment labels. The data is preprocessed by tokenizing the text, building a vocabulary of the most common words, encoding and padding the sequences, and preparing PyTorch tensors and data loaders for efficient batch processing.

A simple neural network model with an embedding layer is defined and trained to classify the sentiment of input texts. The model is trained using cross-entropy loss and the Adam optimizer. Performance is evaluated on a test set using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix, providing insights into the model's effectiveness across all sentiment classes.

# Tools & Libraries

Python: Primary programming language.
Jupyter Notebook: Interactive environment for code, visualization, and explanation.
NumPy & Pandas: Data manipulation and analysis.
PyTorch: Model definition, training, and evaluation.
Torchtext/DataLoader: Efficient batch processing of text data.
Scikit-learn: Calculation of classification metrics and confusion matrix.
Hugging Face Datasets: Source for the labeled multiclass sentiment dataset.
