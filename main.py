# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:49:40 2021

@author: he
"""
#importing neccessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

df = pd.read_csv('final_news_data.csv')

news_classes = df.label

news_classes.head()

#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['text'], news_classes, test_size=0.2, random_state=42)

#initializing the Vectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.7)

#fit and transform the train and transform the test

tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

#initializing PassiveAggressiveClassifier & setting n_epochs
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

#predicting on the test and calculate the accuracy.

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy : {round(score*100, 2)}%')

#Building the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

conf_matrix_table = classification_report(y_test, y_pred, labels=['FAKE', 'REAL'])

#visualization of confusion matrix

group_names = ['True Negative','False Positive','False Negative','True Positive']

group_counts = ['{0:0.0f}'.format(value) for value in
                conf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
