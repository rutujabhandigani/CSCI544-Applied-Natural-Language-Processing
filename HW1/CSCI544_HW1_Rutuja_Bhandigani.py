#!/usr/bin/env python
# coding: utf-8

#version: Python 3
#! pip install bs4 # in case you don't have it installed
#! pip install contractions

import pandas as pd
import numpy as np
import nltk


import re
from bs4 import BeautifulSoup
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')

from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings("ignore")

'''
References:
https://medium.com/analytics-vidhya/pandas-how-to-change-value-based-on-condition-fc8ee38ba529
https://stackoverflow.com/questions/52507752/randomly-select-a-row-from-each-group-using-pandas
https://stackoverflow.com/questions/51994254/removing-url-from-a-column-in-pandas-dataframe
https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element
https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/
https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
https://stackoverflow.com/questions/47557563/lemmatization-of-all-pandas-cells
https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
'''


#read the dataset
data = pd.read_csv("amazon_reviews_us_Beauty_v1_00.tsv", sep = '\t', on_bad_lines='skip')

#Keep only required columns (reviews and ratings)
data = data[["review_body", "star_rating"]]
#data.isnull().sum()

#drop null and NaN values
data.dropna(inplace = True)

#convert ratings to int
data = data.astype({'star_rating': 'int'})

#create 3 classes based on ratings
rating_conditions = [
    (data['star_rating'] <= 2),
    (data['star_rating'] > 2) & (data['star_rating'] < 4),
    (data['star_rating'] > 3)
]
rating_class = [1, 2, 3]
data['class'] = np.select(rating_conditions, rating_class)

#select 20,000 random reviews from each class to create a balanced dataset
np.random.seed(0)
data = data.groupby(['class'])['review_body'].apply(pd.Series.sample, n=20000).reset_index(level=[0, 1])

#average length of reviews before cleaning
data['review_length'] = data['review_body'].str.len()
review_len_before_cleaning = data['review_length'].mean()

# Convert all reviews to lowercase
data['review_body'] = data['review_body'].str.lower()

# Remove HTML and URLs from the reviews
data['review_body'] = data['review_body'].apply(lambda x: re.sub(r'(<.*?>|https?://\S+)', '', x))

# remove non-alphabetical characters
data['review_body'] = data['review_body'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

# remove extra spaces
data['review_body'] = data['review_body'].str.strip()

# Perform contractions on the reviews
data['review_body'] = data['review_body'].apply(lambda x: contractions.fix(x))

# Print average length of reviews before and after cleaning
review_lengths = data['review_body'].str.len()
review_len_after_cleaning = review_lengths.mean()
print("Average review length before and after cleaning:", review_len_before_cleaning,",", review_len_after_cleaning)

# remove stopwords
stopwords_list = stopwords.words('english')
data['review_body'] = data['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_list]))

#perform lemmatization
lemmatizer = WordNetLemmatizer()
data['review_body'] = data['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

# Print average length of reviews before and after preprocessing
review_lengths = data['review_body'].str.len()
review_len_after_preprocessing = review_lengths.mean()
print("Average review length before and after preprocessing:", review_len_after_cleaning, ",", review_len_after_preprocessing)

# create the tf-idf feature matrix
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(data['review_body'])

# split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(features, data['class'], stratify= data['class'], test_size=0.2)


def perceptron_model():
    # Initialize the Perceptron model
    clf = Perceptron(tol=1e-3, random_state=42)

    # Train the model on the training dataset
    clf.fit(X_train, y_train)

    # Make predictions on the testing dataset
    y_pred = clf.predict(X_test)

    # Compute the precision, recall, and f1-score per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # Compute the average precision, recall, and f1-score
    average_precision = precision.mean()
    average_recall = recall.mean()
    average_f1 = f1.mean()

    # Compute the accuracy
    acc = accuracy_score(y_test, y_pred)
    
    print('-'*30)
    print('Perceptron Model metrics: \n')
    
    # Print the precision, recall, f1-score per class
    print("Class 1:", precision[0],",", recall[0],",", f1[0])
    print("Class 2:", precision[1],",", recall[1],",", f1[1])
    print("Class 3:", precision[2],",", recall[2],",", f1[2])

    # Print the average precision, recall, f1-score
    print("Average of each metric:", average_precision, ",", average_recall, ",", average_f1)
    
    
def svm_model():

    # Create an instance of the LinearSVC classifier
    svc = LinearSVC()

    # Define the parameter grid to search
    param_grid = {'C': [0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']}

    # Create an instance of the GridSearchCV class
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Re-train the model with the best parameters
    svc = LinearSVC(**best_params)
    svc.fit(X_train, y_train)

    # Make predictions on the testing dataset
    y_pred = svc.predict(X_test)

    # Compute the precision, recall, and f1-score per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # Compute the average precision, recall, and f1-score
    average_precision = precision.mean()
    average_recall = recall.mean()
    average_f1 = f1.mean()

    # Compute the accuracy
    acc = accuracy_score(y_test, y_pred)
    
    print('-'*30)
    print('SVM Model metrics: \n')
    
    # Print the precision, recall, f1-score per class
    print("Class 1:", precision[0],",", recall[0],",", f1[0])
    print("Class 2:", precision[1],",", recall[1],",", f1[1])
    print("Class 3:", precision[2],",", recall[2],",", f1[2])

    # Print the average precision, recall, f1-score
    print("Average of each metric:", average_precision, ",", average_recall, ",", average_f1)
    
def logistic_reg():

    # Initialize the Logistic Regression model
    logreg = LogisticRegression(random_state=42)

    # Train the model on the training dataset
    logreg.fit(X_train, y_train)

    # Make predictions on the testing dataset
    y_pred = logreg.predict(X_test)

    # Compute the precision, recall, and f1-score per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # Compute the average precision, recall, and f1-score
    average_precision = precision.mean()
    average_recall = recall.mean()
    average_f1 = f1.mean()

    # Compute the accuracy
    acc = accuracy_score(y_test, y_pred)
    
    print('-'*30)
    print('Logistic Regression Model metrics: \n')
    
    # Print the precision, recall, f1-score per class
    print("Class 1:", precision[0],",", recall[0],",", f1[0])
    print("Class 2:", precision[1],",", recall[1],",", f1[1])
    print("Class 3:", precision[2],",", recall[2],",", f1[2])

    # Print the average precision, recall, f1-score
    print("Average of each metric:", average_precision, ",", average_recall, ",", average_f1)
    

def naive_bayes_model():

    # train the Multinomial Naive Bayes model on the training dataset
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)

    # predict labels for the testing dataset
    y_pred = mnb.predict(X_test)

    # Compute the precision, recall, and f1-score per class
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # Compute the average precision, recall, and f1-score
    average_precision = precision.mean()
    average_recall = recall.mean()
    average_f1 = f1.mean()

    # Compute the accuracy
    acc = accuracy_score(y_test, y_pred)
    
    print('-'*30)
    print('Naive Bayes Model metrics: \n')
    
    # Print the precision, recall, f1-score per class
    print("Class 1:", precision[0],",", recall[0],",", f1[0])
    print("Class 2:", precision[1],",", recall[1],",", f1[1])
    print("Class 3:", precision[2],",", recall[2],",", f1[2])

    # Print the average precision, recall, f1-score
    print("Average of each metric:", average_precision, ",", average_recall, ",", average_f1)

def main():
    perceptron_model()
    svm_model()
    logistic_reg()
    naive_bayes_model()
    

if __name__ == "__main__":
    main()
