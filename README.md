# Spam Email Classifier

## Overview
This repository contains the code for a spam email classifier implemented in Python using machine learning techniques. The goal of the project is to develop a model that can accurately classify emails as either spam or not spam (ham).

## Features
- **Data Preprocessing**: The project includes preprocessing steps such as lowercasing, tokenization, removal of stopwords and punctuation, and stemming to clean the text data.
- **Feature Extraction**: Text data is transformed into TF-IDF vectors to represent the features for training the classifier.
- **Machine Learning Model**: The classifier is implemented using the Multinomial Naive Bayes algorithm, which is well-suited for text classification tasks.
- **Evaluation**: Model performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix metrics.
- **Testing**: The model can be tested with custom input text to classify emails in real-time.

## Dataset
The dataset used for training and testing the classifier is included in this repository. It contains a collection of emails labeled as spam or ham. You can find the dataset in the `dataset` directory. If you want to use a different dataset, you can easily obtain it from various sources online.

## Usage
1. **Preprocessing**: Run the preprocessing script to clean and preprocess the text data.
2. **Training**: Train the classifier using the preprocessed data.
3. **Testing**: Test the trained model with custom input text to classify emails.

## Dependencies
- Python 3.x
- scikit-learn
- NLTK (Natural Language Toolkit)

## Contribution
Contributions to improve the classifier's performance, add new features, or fix issues are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

